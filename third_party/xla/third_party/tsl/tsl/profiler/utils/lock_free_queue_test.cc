/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tsl/profiler/utils/lock_free_queue.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/synchronization/notification.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"

namespace tsl {
namespace profiler {
namespace {

template <typename T, size_t block_size_in_bytes>
void RetriveEvents(LockFreeQueue<T, block_size_in_bytes>& queue,
                   absl::Notification& stopped, std::vector<T>& result) {
  result.clear();
  do {
    while (auto event = queue.Pop()) {
      result.emplace_back(*event);
    }
  } while (!stopped.HasBeenNotified());
  while (auto event = queue.Pop()) {
    result.emplace_back(*event);
  }
}

template <typename T, size_t block_size_in_bytes, typename Generator>
void FillEvents2Stage(LockFreeQueue<T, block_size_in_bytes>& queue,
                      Generator gen, size_t event_count1, size_t event_count2,
                      absl::Notification& stage1_filled,
                      absl::Notification& stage1_grabbed,
                      absl::Notification& stage2_filled,
                      std::vector<T>& expected1, std::vector<T>& expected2) {
  expected1.clear();
  expected2.clear();
  for (size_t i = 0; i < event_count1; ++i) {
    T event = gen(i);
    expected1.emplace_back(event);
    queue.Push(std::move(event));
  }
  stage1_filled.Notify();
  stage1_grabbed.WaitForNotification();
  for (size_t i = 0; i < event_count2; ++i) {
    T event = gen(i + event_count1);
    expected2.emplace_back(event);
    queue.Push(std::move(event));
  }
  stage2_filled.Notify();
}

template <typename T, size_t block_size_in_bytes, typename Generator>
void TestProducerConsumer(size_t event_count1, size_t event_count2,
                          Generator gen) {
  LockFreeQueue<T, block_size_in_bytes> queue;
  std::vector<T> expected1;
  std::vector<T> expected2;
  absl::Notification stage1_filled;
  absl::Notification stage1_grabbed;
  absl::Notification stage2_filled;

  auto* producer = Env::Default()->StartThread(
      ThreadOptions(), "producer", [&, gen, event_count1, event_count2]() {
        FillEvents2Stage(queue, gen, event_count1, event_count2, stage1_filled,
                         stage1_grabbed, stage2_filled, expected1, expected2);
      });

  std::vector<T> result1;
  auto* consumer1 = Env::Default()->StartThread(
      ThreadOptions(), "consumer1", [&queue, &result1, &stage1_filled]() {
        RetriveEvents(queue, stage1_filled, result1);
      });

  delete consumer1;
  EXPECT_THAT(result1, ::testing::ContainerEq(expected1));
  stage1_grabbed.Notify();

  std::vector<T> result2;
  auto* consumer2 = Env::Default()->StartThread(
      ThreadOptions(), "consumer2", [&queue, &result2, &stage2_filled]() {
        RetriveEvents(queue, stage2_filled, result2);
      });
  delete consumer2;
  EXPECT_THAT(result2, ::testing::ContainerEq(expected2));

  delete producer;
}

template <typename T, size_t block_size_in_bytes, typename Generator,
          bool producer_concurrent>
void TestGrabFrom(size_t event_count1, size_t event_count2, Generator gen) {
  LockFreeQueue<T, block_size_in_bytes> queue;
  std::vector<T> result;
  std::vector<T> expected1;
  std::vector<T> expected2;
  absl::Notification stage1_filled;
  absl::Notification stage1_grabbed;
  absl::Notification stage2_filled;

  auto* producer = Env::Default()->StartThread(
      ThreadOptions(), "producer", [&, gen, event_count1, event_count2]() {
        FillEvents2Stage(queue, gen, event_count1, event_count2, stage1_filled,
                         stage1_grabbed, stage2_filled, expected1, expected2);
      });

  stage1_filled.WaitForNotification();
  LockFreeQueue<T, block_size_in_bytes> dumped_queue1;
  dumped_queue1.GrabFrom(queue, producer_concurrent);
  std::vector<T> result1;
  while (auto event = dumped_queue1.Pop()) {
    result1.emplace_back(*event);
  }
  EXPECT_THAT(result1, ::testing::ContainerEq(expected1));

  stage1_grabbed.Notify();
  delete producer;
  LockFreeQueue<T, block_size_in_bytes> dumped_queue2;
  dumped_queue2.GrabFrom(queue, producer_concurrent);
  std::vector<T> result2;
  while (auto event = dumped_queue2.Pop()) {
    result2.emplace_back(*event);
  }
  EXPECT_THAT(result2, ::testing::ContainerEq(expected2));
}

template <typename T, size_t block_size_in_bytes, typename Generator>
void TestIterator(size_t event_count1, size_t event_count2, Generator gen) {
  LockFreeQueue<T, block_size_in_bytes> queue;
  std::vector<T> result;
  std::vector<T> expected1;
  std::vector<T> expected2;
  absl::Notification stage1_filled;
  absl::Notification stage1_grabbed;
  absl::Notification stage2_filled;

  auto* producer = Env::Default()->StartThread(
      ThreadOptions(), "producer", [&, gen, event_count1, event_count2]() {
        FillEvents2Stage(queue, gen, event_count1, event_count2, stage1_filled,
                         stage1_grabbed, stage2_filled, expected1, expected2);
      });

  auto iterate_and_collect = [&queue](std::vector<T>& events) {
    events.clear();
    for (auto it = queue.begin(), ite = queue.end(); it != ite; ++it) {
      events.emplace_back(*it);
    }
    queue.Clear();
  };

  std::vector<T> result1;
  stage1_filled.WaitForNotification();
  iterate_and_collect(result1);
  EXPECT_THAT(result1, ::testing::ContainerEq(expected1));
  stage1_grabbed.Notify();

  std::vector<T> result2;
  stage2_filled.WaitForNotification();
  iterate_and_collect(result2);
  EXPECT_THAT(result2, ::testing::ContainerEq(expected2));

  delete producer;
}

TEST(LockFreeQueueTest, Int64Event_ProducerConsumer) {
  auto gen = [](size_t i) -> int64_t { return static_cast<int64_t>(i); };
  using T = decltype(gen(0));
  constexpr size_t kBS = 512;
  using G = decltype(gen);
  constexpr size_t kNumSlots = LockFreeQueue<T, kBS>::kNumSlotsPerBlock;
  EXPECT_GE(kNumSlots, 10);

  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, 2, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, 3, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, 5, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, kNumSlots + 3, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, kNumSlots * 2 + 5, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots, 2, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots, kNumSlots - 1, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots, kNumSlots, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots, kNumSlots + 1, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 2, kNumSlots + 1, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 3, kNumSlots - 1, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots + 3, 2, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots + 3, kNumSlots - 4, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots + 3, kNumSlots - 3, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots + 3, kNumSlots - 2, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots - 5, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots + 3, gen);
}

TEST(LockFreeQueueTest, StringEvent_ProducerConsumer) {
  auto gen = [](size_t i) { return std::to_string(i); };
  using T = decltype(gen(0));
  constexpr size_t kBS = 512;
  using G = decltype(gen);
  constexpr size_t kNumSlots = LockFreeQueue<T, kBS>::kNumSlotsPerBlock;
  EXPECT_GE(kNumSlots, 10);

  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, 2, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, 3, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, 5, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, kNumSlots + 3, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots - 3, kNumSlots * 2 + 5, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots, 2, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots, kNumSlots - 1, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots, kNumSlots, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots, kNumSlots + 1, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 2, kNumSlots + 1, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 3, kNumSlots - 1, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots + 3, 2, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots + 3, kNumSlots - 4, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots + 3, kNumSlots - 3, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots + 3, kNumSlots - 2, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots - 5, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots, gen);
  TestProducerConsumer<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots + 3, gen);
}

TEST(LockFreeQueueTest, Int64Event_GrabFrom) {
  auto gen = [](size_t i) -> int64_t { return static_cast<int64_t>(i); };
  using T = decltype(gen(0));
  constexpr size_t kBS = 512;
  using G = decltype(gen);
  constexpr size_t kNumSlots = LockFreeQueue<T, kBS>::kNumSlotsPerBlock;
  EXPECT_GE(kNumSlots, 10);

  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, 2, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, 3, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, 5, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, kNumSlots + 3, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, kNumSlots * 2 + 5, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots, 2, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots, kNumSlots - 1, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots, kNumSlots, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots, kNumSlots + 1, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 2, kNumSlots + 1, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 3, kNumSlots - 1, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots + 3, 2, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots + 3, kNumSlots - 4, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots + 3, kNumSlots - 3, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots + 3, kNumSlots - 2, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 2 + 3, kNumSlots - 5, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 2 + 3, kNumSlots, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 2 + 3, kNumSlots + 3, gen);
}

TEST(LockFreeQueueTest, Int64Event_GrabFrom_no_producer_concurrent) {
  auto gen = [](size_t i) -> int64_t { return static_cast<int64_t>(i); };
  using T = decltype(gen(0));
  constexpr size_t kBS = 512;
  using G = decltype(gen);
  constexpr size_t kNumSlots = LockFreeQueue<T, kBS>::kNumSlotsPerBlock;
  EXPECT_GE(kNumSlots, 10);

  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, 2, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, 3, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, 5, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, kNumSlots + 3, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, kNumSlots * 2 + 5, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots, 2, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots, kNumSlots - 1, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots, kNumSlots, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots, kNumSlots + 1, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 2, kNumSlots + 1, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 3, kNumSlots - 1, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots + 3, 2, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots + 3, kNumSlots - 4, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots + 3, kNumSlots - 3, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots + 3, kNumSlots - 2, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 2 + 3, kNumSlots - 5, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 2 + 3, kNumSlots, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 2 + 3, kNumSlots + 3, gen);
}

TEST(LockFreeQueueTest, StringEvent_GrabFrom) {
  auto gen = [](size_t i) -> std::string { return std::to_string(i); };
  using T = decltype(gen(0));
  constexpr size_t kBS = 512;
  using G = decltype(gen);
  constexpr size_t kNumSlots = LockFreeQueue<T, kBS>::kNumSlotsPerBlock;
  EXPECT_GE(kNumSlots, 10);

  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, 2, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, 3, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, 5, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, kNumSlots + 3, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots - 3, kNumSlots * 2 + 5, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots, 2, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots, kNumSlots - 1, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots, kNumSlots, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots, kNumSlots + 1, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 2, kNumSlots + 1, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 3, kNumSlots - 1, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots + 3, 2, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots + 3, kNumSlots - 4, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots + 3, kNumSlots - 3, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots + 3, kNumSlots - 2, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 2 + 3, kNumSlots - 5, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 2 + 3, kNumSlots, gen);
  TestGrabFrom<T, kBS, G, true>(kNumSlots * 2 + 3, kNumSlots + 3, gen);
}

TEST(LockFreeQueueTest, StringEvent_GrabFrom_no_producer_concurrent) {
  auto gen = [](size_t i) -> std::string { return std::to_string(i); };
  using T = decltype(gen(0));
  constexpr size_t kBS = 512;
  using G = decltype(gen);
  constexpr size_t kNumSlots = LockFreeQueue<T, kBS>::kNumSlotsPerBlock;
  EXPECT_GE(kNumSlots, 10);

  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, 2, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, 3, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, 5, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, kNumSlots + 3, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots - 3, kNumSlots * 2 + 5, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots, 2, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots, kNumSlots - 1, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots, kNumSlots, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots, kNumSlots + 1, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 2, kNumSlots + 1, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 3, kNumSlots - 1, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots + 3, 2, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots + 3, kNumSlots - 4, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots + 3, kNumSlots - 3, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots + 3, kNumSlots - 2, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 2 + 3, kNumSlots - 5, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 2 + 3, kNumSlots, gen);
  TestGrabFrom<T, kBS, G, false>(kNumSlots * 2 + 3, kNumSlots + 3, gen);
}

TEST(LockFreeQueueTest, Int64Event_Iterator) {
  auto gen = [](size_t i) -> int64_t { return static_cast<int64_t>(i); };
  using T = decltype(gen(0));
  constexpr size_t kBS = 512;
  using G = decltype(gen);
  constexpr size_t kNumSlots = LockFreeQueue<T, kBS>::kNumSlotsPerBlock;
  EXPECT_GE(kNumSlots, 10);

  TestIterator<T, kBS, G>(kNumSlots - 3, 2, gen);
  TestIterator<T, kBS, G>(kNumSlots - 3, 3, gen);
  TestIterator<T, kBS, G>(kNumSlots - 3, 5, gen);
  TestIterator<T, kBS, G>(kNumSlots - 3, kNumSlots + 3, gen);
  TestIterator<T, kBS, G>(kNumSlots - 3, kNumSlots * 2 + 5, gen);
  TestIterator<T, kBS, G>(kNumSlots, 2, gen);
  TestIterator<T, kBS, G>(kNumSlots, kNumSlots - 1, gen);
  TestIterator<T, kBS, G>(kNumSlots, kNumSlots, gen);
  TestIterator<T, kBS, G>(kNumSlots, kNumSlots + 1, gen);
  TestIterator<T, kBS, G>(kNumSlots * 2, kNumSlots + 1, gen);
  TestIterator<T, kBS, G>(kNumSlots * 3, kNumSlots - 1, gen);
  TestIterator<T, kBS, G>(kNumSlots + 3, 2, gen);
  TestIterator<T, kBS, G>(kNumSlots + 3, kNumSlots - 4, gen);
  TestIterator<T, kBS, G>(kNumSlots + 3, kNumSlots - 3, gen);
  TestIterator<T, kBS, G>(kNumSlots + 3, kNumSlots - 2, gen);
  TestIterator<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots - 5, gen);
  TestIterator<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots, gen);
  TestIterator<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots + 3, gen);
}

TEST(LockFreeQueueTest, StringEvent_Iterator) {
  auto gen = [](size_t i) { return std::to_string(i); };
  using T = decltype(gen(0));
  constexpr size_t kBS = 512;
  using G = decltype(gen);
  constexpr size_t kNumSlots = LockFreeQueue<T, kBS>::kNumSlotsPerBlock;
  EXPECT_GE(kNumSlots, 10);

  TestIterator<T, kBS, G>(kNumSlots - 3, 2, gen);
  TestIterator<T, kBS, G>(kNumSlots - 3, 3, gen);
  TestIterator<T, kBS, G>(kNumSlots - 3, 5, gen);
  TestIterator<T, kBS, G>(kNumSlots - 3, kNumSlots + 3, gen);
  TestIterator<T, kBS, G>(kNumSlots - 3, kNumSlots * 2 + 5, gen);
  TestIterator<T, kBS, G>(kNumSlots, 2, gen);
  TestIterator<T, kBS, G>(kNumSlots, kNumSlots - 1, gen);
  TestIterator<T, kBS, G>(kNumSlots, kNumSlots, gen);
  TestIterator<T, kBS, G>(kNumSlots, kNumSlots + 1, gen);
  TestIterator<T, kBS, G>(kNumSlots * 2, kNumSlots + 1, gen);
  TestIterator<T, kBS, G>(kNumSlots * 3, kNumSlots - 1, gen);
  TestIterator<T, kBS, G>(kNumSlots + 3, 2, gen);
  TestIterator<T, kBS, G>(kNumSlots + 3, kNumSlots - 4, gen);
  TestIterator<T, kBS, G>(kNumSlots + 3, kNumSlots - 3, gen);
  TestIterator<T, kBS, G>(kNumSlots + 3, kNumSlots - 2, gen);
  TestIterator<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots - 5, gen);
  TestIterator<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots, gen);
  TestIterator<T, kBS, G>(kNumSlots * 2 + 3, kNumSlots + 3, gen);
}

}  // namespace
}  // namespace profiler
}  // namespace tsl
