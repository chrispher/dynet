#include "dynet/init.h"
#include "dynet/aligned-mem-pool.h"
#include "dynet/dynet.h"
#include "dynet/weight-decay.h"
#include "dynet/globals.h"
#include "dynet/str-util.h"

#include <iostream>
#include <random>
#include <cmath>

using namespace std;

namespace dynet {

DynetParams::DynetParams() : random_seed(0), mem_descriptor("512"), weight_decay(0), autobatch(0), profiling(0),
  shared_parameters(false), ngpus_requested(false), ids_requested(false), cpu_requested(false), requested_gpus(-1)
{
}

DynetParams::~DynetParams()
{
}

static void remove_args(int& argc, char**& argv, int& argi, int n) {
  for (int i = argi + n; i < argc; ++i)
    argv[i - n] = argv[i];
  argc -= n;
  DYNET_ASSERT(argc >= 0, "remove_args less than 0");
}

DynetParams extract_dynet_params(int& argc, char**& argv, bool shared_parameters) {
  DynetParams params;
  params.shared_parameters = shared_parameters;

  int argi = 1;
  while (argi < argc) {
    string arg = argv[argi];

    // Memory
    if (arg == "--dynet-mem" || arg == "--dynet_mem") {
      if ((argi + 1) >= argc) {
        throw std::invalid_argument("[dynet] --dynet-mem expects an argument (the memory, in megabytes, to reserve)");
      } else {
        params.mem_descriptor = argv[argi + 1];
        remove_args(argc, argv, argi, 2);
      }
    }

    // Weight decay
    else if (arg == "--dynet-weight-decay" || arg == "--dynet_weight_decay") {
      if ((argi + 1) >= argc) {
        throw std::invalid_argument("[dynet] --dynet-weight-decay requires an argument (the weight decay per update)");
      } else {
        string a2 = argv[argi + 1];
        istringstream d(a2); d >> params.weight_decay;
        remove_args(argc, argv, argi, 2);
      }
    }

    // Random seed
    else if (arg == "--dynet-seed" || arg == "--dynet_seed") {
      if ((argi + 1) >= argc) {
        throw std::invalid_argument("[dynet] --dynet-seed expects an argument (the random number seed)");
      } else {
        string a2 = argv[argi + 1];
        istringstream c(a2); c >> params.random_seed;
        remove_args(argc, argv, argi, 2);
      }
    }

    // Memory
    else if (arg == "--dynet-autobatch" || arg == "--dynet_autobatch") {
      if ((argi + 1) >= argc) {
        throw std::invalid_argument("[dynet] --dynet-autobatch expects an argument (0 for none 1 for on)");
      } else {
        string a2 = argv[argi + 1];
        istringstream c(a2); c >> params.autobatch;
        remove_args(argc, argv, argi, 2);
      }
    }
    else if (arg == "--dynet-profiling" || arg == "--dynet_profiling") {
      string a2 = argv[argi + 1];
      istringstream c(a2); c >> params.profiling;
      remove_args(argc, argv, argi, 2);
    }

    else if (arg == "--dynet-devices" || arg == "--dynet_devices") {
      if ((argi + 1) >= argc) {
        throw std::invalid_argument("[dynet] --dynet-devices expects an argument (comma separated list of CPU and physical GPU ids to use)");
      } else {
        string devices_str = argv[argi + 1];
        if (params.ids_requested)
           throw std::invalid_argument("Multiple instances of --dynet-devices");
        params.ids_requested = true;
        auto devices_info_lst = str_split(devices_str, ',');
        for (auto & devices_info : devices_info_lst) {
          if (startswith(devices_info, "CPU:")) {
            throw std::invalid_argument("DyNet doesn't support specifying CPU id");
          } else if (startswith(devices_info, "CPU")) {
            if (params.cpu_requested)
              throw std::invalid_argument("Bad argument to --dynet-devices");
            params.cpu_requested = true;
          } else if (startswith(devices_info, "GPU:")) {
            throw std::runtime_error("DyNet not support GPU.");
          } else {
            throw std::invalid_argument("Bad argument to --dynet-devices");
          }
        }
        params.cpu_requested = true;
        remove_args(argc, argv, argi, 2);
      }
    }

    // Go to next argument
    else {
      argi++;
    }
  }
  return params;
}

void initialize(DynetParams& params) {
  if (default_device != nullptr) {
    cerr << "WARNING: Attempting to initialize dynet twice. Ignoring duplicate initialization." << endl;
    return;
  }

  DeviceManager* device_manager = get_device_manager();

  // Set random seed
  if (params.random_seed == 0) {
    random_device rd;
    params.random_seed = rd();
  }
  cerr << "[dynet] random seed: " << params.random_seed << endl;
  rndeng = new mt19937(params.random_seed);

  // Set weight decay rate
  if (params.weight_decay < 0 || params.weight_decay >= 1)
    throw std::invalid_argument("[dynet] weight decay parameter must be between 0 and 1 (probably very small like 1e-6)\n");
  weight_decay_lambda = params.weight_decay;

  // Set autobatch
  if(params.autobatch)
    cerr << "[dynet] using autobatching" << endl;
  autobatch_flag = params.autobatch;
  
  if(params.profiling)
    cerr << "[dynet] using profiling level " << params.profiling << endl;
  profiling_flag = params.profiling;

  // Allocate memory
  cerr << "[dynet] allocating memory: " << params.mem_descriptor << "MB\n";
  int default_index = 0;

  Device *d;
  d = new Device_CPU(device_manager->num_devices(), params.mem_descriptor, params.shared_parameters);
  device_manager->add(d);
  default_device = device_manager->get(default_index);

  // TODO these should be accessed through the relevant device and removed here
  kSCALAR_MINUSONE = default_device->kSCALAR_MINUSONE;
  kSCALAR_ONE = default_device->kSCALAR_ONE;
  kSCALAR_ZERO = default_device->kSCALAR_ZERO;
  cerr << "[dynet] memory allocation done.\n";

}

void initialize(int& argc, char**& argv, bool shared_parameters) {
  DynetParams params = extract_dynet_params(argc, argv, shared_parameters);
  initialize(params);
}

void cleanup() {
  delete rndeng;
  get_device_manager()->clear();
  default_device = nullptr;
}

} // namespace dynet
