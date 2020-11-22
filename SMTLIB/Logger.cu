//===----------------------------------------------------------------------===//
//
//                                     JFS
//
// Copyright 2017-2018 Daniel Liew
//
// This file is distributed under the MIT license.
// See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

// BDG: Most of this file has been no-opped in the process of converting
//      to CUDA. We can reintroduce some of it when I figure out how to
//      get std::unordered_map to work in CUDA.

#include "Logger.h"
#include "jassert.h"
#include <cstdio>
#include <cstdlib>
#include <inttypes.h>
#include <string>
#include <unordered_map>

namespace {

class LoggerImpl {
private:
  //std::string logPath;
  //std::unordered_map<const char*, uint64_t> stats;

public:
  __device__ LoggerImpl(const char* path) {}
  __device__ ~LoggerImpl() {
    // Log stats to file.
    // TODO: Rewrite using std classes
    //for (const auto& pair : stats) {
    //  printf("%s: %" PRIu64 "\n", pair.first, pair.second);
    //}
  }
  __device__ void log_uint64(const char* name, uint64_t value) {
    jassert(name != nullptr);
    //stats[name] = value;
  }
};

} // namespace

extern "C" {
__device__ jfs_nr_logger_ty jfs_nr_mk_logger(const char* path) {
  LoggerImpl* l = new LoggerImpl(path);
  return reinterpret_cast<jfs_nr_logger_ty>(l);
}

__device__ jfs_nr_logger_ty jfs_nr_mk_logger_from_env() {
  return jfs_nr_mk_logger("env");
}

__device__ void jfs_nr_log_uint64(jfs_nr_logger_ty logger, const char* name,
                       uint64_t value) {
  LoggerImpl* l = (LoggerImpl*)logger;
  l->log_uint64(name, value);
}

__device__ void jfs_nr_del_logger(jfs_nr_logger_ty logger) {
  //LoggerImpl* l = (LoggerImpl*)logger;
  //delete l;
}
}
