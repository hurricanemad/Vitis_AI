/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/vitis/ai/reidtracker.hpp"
#include "tracker_imp.hpp"

namespace vitis {
namespace ai {

ReidTracker::ReidTracker() {}
ReidTracker::~ReidTracker() {}

std::shared_ptr<ReidTracker> ReidTracker::create(uint64_t mode,
                                                 const SpecifiedCfg &cfg) {
  return std::shared_ptr<ReidTracker>(new ReidTrackerImp(mode, cfg));
}

}  // namespace ai
}  // namespace vitis
