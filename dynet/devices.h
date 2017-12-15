#ifndef DYNET_DEVICES_H
#define DYNET_DEVICES_H

#include <unordered_map>
#include <string>
#include <exception>
#include "dynet/aligned-mem-pool.h"
#include "dynet/globals.h"

namespace Eigen {
  struct DefaultDevice;
}

namespace dynet {

enum class DeviceType {CPU};

/*
 * FXS   -> forward pass memory
 * DEDFS -> backward pass memory
 * PS    -> parameter memory
 * SCS   -> scratch memory (for use in temporary calculations)
 * NONE  -> when a memory pool has not been assigned yet
 */
enum class DeviceMempool {FXS = 0, DEDFS = 1, PS = 2, SCS = 3, NONE = 4};

struct ComputationGraph; // TODO is there a nicer way to resolve this cyclic dependency?
struct Tensor;

struct DeviceMempoolSizes {
  size_t used[4];
  DeviceMempoolSizes() = default;
  DeviceMempoolSizes(size_t total_s);
  DeviceMempoolSizes(size_t fxs_s, size_t dEdfs_s, size_t ps_s, size_t sc_s);
  DeviceMempoolSizes(const std::string & descriptor);
};


class Device {
 protected:
  Device(int i, DeviceType t, MemAllocator* m) : device_id(i), type(t), mem(m), pools(4, nullptr) {}
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  virtual ~Device();
 public:
  int device_id;
  DeviceType type;
  MemAllocator* mem;
  float* kSCALAR_MINUSONE;
  float* kSCALAR_ONE;
  float* kSCALAR_ZERO;
  std::string name;
  virtual DeviceMempoolSizes mark(ComputationGraph *cg);
  virtual void revert(const DeviceMempoolSizes & cp);
  void allocate_tensor(DeviceMempool mem_pool, Tensor & tensor);
  std::vector<AlignedMemoryPool*> pools;
};

class Device_CPU : public Device {
 public:
  typedef Eigen::DefaultDevice EigenDevice;
  explicit Device_CPU(int my_id, const DeviceMempoolSizes & mb, bool shared);
  ~Device_CPU();
  CPUAllocator cpu_mem;
  Eigen::DefaultDevice* edevice;
  MemAllocator* shmem;
};

class DeviceManager final {
 public:
  DeviceManager();
  ~DeviceManager();

  void clear();

  void add(Device* d);

  Device* get(size_t i) { return devices[i]; }

  size_t num_devices() const { return devices.size(); }

  const std::vector<Device*>& get_devices() const { return devices; }

  Device* get_global_device(const std::string & name);

  // no copying allowed
  DeviceManager(const DeviceManager &) = delete;
  void operator=(const DeviceManager &) = delete;

 private:
  std::vector<Device*> devices;
  std::unordered_map<std::string, Device*> devices_map;
};

DeviceManager* get_device_manager();

} // namespace dynet

#endif
