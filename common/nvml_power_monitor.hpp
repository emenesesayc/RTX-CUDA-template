#ifndef NVML_POWER_MONITOR_HPP
#define NVML_POWER_MONITOR_HPP

/*
  nvml_power_monitor.hpp — header-only NVML GPU power monitor
  --------------------------------------------------------------------
  Units:
    nvmlDeviceGetPowerUsage() returns milliwatts (mW)
    We convert to watts (W) for power.
    Energy is accumulated as Joules (J) via power (W) * dt (s).
    Time is seconds (s).
  --------------------------------------------------------------------
  Usage:
    #include "nvml_power_monitor.hpp"
    ...
    GPUPowerBegin("myalg", 100); // sample every 100 ms
    ... run workload ...
    GPUPowerEnd();
  --------------------------------------------------------------------
  Link with: -lnvidia-ml -lpthread
    Example:
      g++ main.cpp -o main -lnvidia-ml -lpthread
  --------------------------------------------------------------------
  Note: NVML must be available on the system (NVIDIA driver installed).
*/

#include <nvml.h>
#include <pthread.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/time.h>
#include <unistd.h>

// ========== Global state (header-only static) ==========
static std::atomic<bool> GPUpollThreadStatus{false};
static unsigned int nvmlDeviceCount = 0;
static nvmlDevice_t nvmlDeviceID = nullptr;
static nvmlPciInfo_t nvmlPCIInfo;
static char nvmlDeviceName[128] = {0};
static std::string GPUfilename = "power.dat";

static int GPU_SAMPLE_MS = 100; // default sample period (ms)

static double gpuCurrentPower = 0.0;   // W (instantaneous)
static double gpuAveragePower = 0.0;   // W (average over measurement)
static double gpuTotalEnergy   = 0.0;  // J (accumulated)
static double gpuTotalTime     = 0.0;  // s (accumulated)

static pthread_t GPUPowerThread;

// Helper: convert timeval to seconds (double)
static inline double tv_to_sec(const struct timeval &t) {
    return double(t.tv_sec) + double(t.tv_usec) / 1e6;
}

// Simple NVML error to integer mapping (keeps parity with your original)
static inline int getNVMLErrorCode(nvmlReturn_t r) {
    if (r == NVML_ERROR_UNINITIALIZED) return 1;
    if (r == NVML_ERROR_INVALID_ARGUMENT) return 2;
    if (r == NVML_ERROR_NOT_SUPPORTED) return 3;
    if (r == NVML_ERROR_NO_PERMISSION) return 4;
    if (r == NVML_ERROR_ALREADY_INITIALIZED) return 5;
    if (r == NVML_ERROR_NOT_FOUND) return 6;
    if (r == NVML_ERROR_INSUFFICIENT_SIZE) return 7;
    if (r == NVML_ERROR_INSUFFICIENT_POWER) return 8;
    if (r == NVML_ERROR_DRIVER_NOT_LOADED) return 9;
    if (r == NVML_ERROR_TIMEOUT) return 10;
    if (r == NVML_ERROR_IRQ_ISSUE) return 11;
    if (r == NVML_ERROR_LIBRARY_NOT_FOUND) return 12;
    if (r == NVML_ERROR_FUNCTION_NOT_FOUND) return 13;
    if (r == NVML_ERROR_CORRUPTED_INFOROM) return 14;
    if (r == NVML_ERROR_GPU_IS_LOST) return 15;
    if (r == NVML_ERROR_UNKNOWN) return 16;
    return 0;
}

// Thread function: polls NVML for power usage
static void* GPUPowerPollingFunc(void* /*arg*/) {
    // Open file and write header (units)
    FILE* fp = fopen(GPUfilename.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "GPUPower: cannot open file %s for writing\n", GPUfilename.c_str());
        pthread_exit(nullptr);
    }

    // Column header with units
    // Columns: timestep, power (W), acc-energy (J), avg-power (W), dt (s), acc-time (s)
    fprintf(fp, "%-12s %-12s %-14s %-12s %-10s %-12s\n",
            "#timestep", "power(W)", "acc-energy(J)", "avg-power(W)", "dt(s)", "acc-time(s)");
    fflush(fp);

    int timestep = 0;
    struct timeval t_prev, t_now;
    gettimeofday(&t_prev, nullptr);

    double acc_energy = 0.0; // J
    double acc_time = 0.0;   // s

    // Sampling loop
    while (GPUpollThreadStatus.load()) {
        // Sleep for sample period
        usleep((useconds_t)GPU_SAMPLE_MS * 1000);

        // Disable cancel while performing critical NVML calls (matching your pattern)
        pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, nullptr);

        timestep++;
        gettimeofday(&t_now, nullptr);
        double dt = tv_to_sec(t_now) - tv_to_sec(t_prev);
        if (dt <= 0.0) dt = 1e-9; // safety

        // Query power usage in milliwatts (mW)
        unsigned int power_mw = 0;
        nvmlReturn_t r = nvmlDeviceGetPowerUsage(nvmlDeviceID, &power_mw);
        if (NVML_SUCCESS != r) {
            // On error, print once and stop thread
            fprintf(stderr, "GPUPower: nvmlDeviceGetPowerUsage error: %s\n", nvmlErrorString(r));
            // Let caller stop normally; set current power to 0 for safety
            gpuCurrentPower = 0.0;
            // still update time counters to avoid division by zero
            acc_time += dt;
            gpuTotalTime = acc_time;
            pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
            continue;
        }

        // Convert to W
        double power_w = double(power_mw) / 1000.0;

        // Accumulate energy and time
        acc_energy += power_w * dt; // J
        acc_time += dt;             // s

        // Update globals
        gpuCurrentPower = power_w;
        gpuTotalEnergy = acc_energy;
        gpuTotalTime = acc_time;
        gpuAveragePower = (acc_time > 0.0) ? (acc_energy / acc_time) : 0.0;

        // Write row: timestep power(J/s) acc-energy(J) avg-power(W) dt(s) acc-time(s)
        fprintf(fp, "%-12d %-12.6f %-14.6f %-12.6f %-10.6f %-12.6f\n",
                timestep, power_w, acc_energy, gpuAveragePower, dt, acc_time);
        fflush(fp);

        // Console update (overwrite)
        //printf("\r[GPU = %10.5f W] ", power_w);
        //fflush(stdout);

        // Prepare for next iteration
        t_prev = t_now;

        pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
    }

    fclose(fp);
    pthread_exit(nullptr);
    return nullptr;
}

// ========== Public API ==========

// Start GPU power measurement
// alg          : algorithm name prefix for the output file -> "power-<alg>.dat"
// ms           : sampling period in milliseconds
// device_index : cuda device index to monitor (matches application-selected GPU)
static inline void GPUPowerBegin(const char* alg, int ms = 100, int device_index = 0) {
    GPU_SAMPLE_MS = ms;
    //GPUfilename = std::string("power-") + std::string(alg) + std::string(".dat");
    GPUfilename = std::string("power.dat");

    // Initialize NVML
    nvmlReturn_t r = nvmlInit();
    if (NVML_SUCCESS != r) {
        fprintf(stderr, "GPUPower: nvmlInit failed: %s\n", nvmlErrorString(r));
        // Caller probably expects the program to exit on NVML not available (like your original).
        // But here we keep it graceful and abort the begin call.
        exit(1);
    }

    // Get device count
    r = nvmlDeviceGetCount(&nvmlDeviceCount);
    if (NVML_SUCCESS != r || nvmlDeviceCount == 0) {
        fprintf(stderr, "GPUPower: nvmlDeviceGetCount failed or zero devices: %s\n", nvmlErrorString(r));
        nvmlShutdown();
        exit(1);
    }

    // Use caller-selected device (defaults to 0 to preserve old behavior).
    if (device_index < 0 || device_index >= nvmlDeviceCount) {
        fprintf(stderr, "GPUPower: invalid device index %d (device_count=%d)\n", device_index, nvmlDeviceCount);
        nvmlShutdown();
        exit(1);
    }
    r = nvmlDeviceGetHandleByIndex(device_index, &nvmlDeviceID);
    if (NVML_SUCCESS != r) {
        fprintf(stderr, "GPUPower: nvmlDeviceGetHandleByIndex(0) failed: %s\n", nvmlErrorString(r));
        nvmlShutdown();
        exit(1);
    }

    // get device name and pci info for possible logging
    if (nvmlDeviceGetName(nvmlDeviceID, nvmlDeviceName, sizeof(nvmlDeviceName)) != NVML_SUCCESS) {
        strncpy(nvmlDeviceName, "unknown", sizeof(nvmlDeviceName));
    }
    (void) nvmlDeviceGetPciInfo(nvmlDeviceID, &nvmlPCIInfo);

    // reset accumulators
    gpuCurrentPower = 0.0;
    gpuAveragePower = 0.0;
    gpuTotalEnergy = 0.0;
    gpuTotalTime = 0.0;

    GPUpollThreadStatus.store(true);

    // spawn thread
    int rc = pthread_create(&GPUPowerThread, nullptr, GPUPowerPollingFunc, nullptr);
    if (rc) {
        fprintf(stderr, "GPUPower: pthread_create returned %d\n", rc);
        GPUpollThreadStatus.store(false);
        nvmlShutdown();
        exit(1);
    }

    // small cooldown (match your pattern)
    usleep(1000000);
}

// Stop GPU power measurement and print summary
static inline double GPUPowerEnd() {
    // small cooldown
    usleep(1000000);

    GPUpollThreadStatus.store(false);
    pthread_join(GPUPowerThread, nullptr);

    // Shutdown NVML
    nvmlReturn_t r = nvmlShutdown();
    if (NVML_SUCCESS != r) {
        fprintf(stderr, "GPUPower: nvmlShutdown failed: %s\n", nvmlErrorString(r));
        // don't exit here; we already have the measurements
    }

    // Print summary (units)
    double ckWh = 3600000.0; // J to kWh conversion: 1 kWh = 3.6e6 J

    printf("\n\n========== GPU POWER SUMMARY ==========\n");
    printf("Device: %s (pci: %s)\n", nvmlDeviceName, nvmlPCIInfo.busId);
    printf("GPU Avg. Power:    %.6f W\n", gpuAveragePower);
    printf("GPU Total Energy:  %.6f J = %.9f kWh\n", gpuTotalEnergy, gpuTotalEnergy / ckWh);
    printf("GPU Total Time:    %.6f s\n", gpuTotalTime);
    printf("=======================================\n\n");
    return gpuTotalEnergy;
}

#endif // NVML_POWER_MONITOR_HPP
