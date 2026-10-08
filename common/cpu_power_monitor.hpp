#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <string>
#include <unistd.h>

// ============================================================
//            RAPL POWER MONITOR (CPU + DRAM)
// ------------------------------------------------------------
// Units:
//   energy_uj: microjoules (µJ)
//   energy J: joules (J)
//   power    : watts (W)          [J/s]
//   time     : seconds (s)
// ============================================================


// ============================================================
//                      RAPL DOMAIN READER
// ============================================================

class RaplDomain {
public:
    std::string energyFile;    // µJ
    std::string maxRangeFile;  // µJ

    double lastEnergyJ = 0;    // J
    double totalEnergyJ = 0;   // J
    double maxEnergyJ = 0;     // J

    RaplDomain(const std::string& path)
        : energyFile(path + "/energy_uj"),
          maxRangeFile(path + "/max_energy_range_uj")
    {
        maxEnergyJ   = readU64(maxRangeFile.c_str()) / 1e6; // µJ → J
        lastEnergyJ  = readU64(energyFile.c_str()) / 1e6;   // µJ → J
    }

    static uint64_t readU64(const char* path) {
        FILE* fp = fopen(path, "r");
        if (!fp) return 0;
        unsigned long long v = 0;
        fscanf(fp, "%llu", &v);
        fclose(fp);
        return v;
    }

    // sample() returns instantaneous delta energy (J) via deltaJ
    double sample(double &deltaJ) {
        double e = readU64(energyFile.c_str()) / 1e6; // J

        double d = e - lastEnergyJ;
        if (d < 0)
            d += maxEnergyJ;  // wraparound correction

        deltaJ = d;
        totalEnergyJ += d;
        lastEnergyJ = e;

        return e;
    }
};


// ============================================================
//                     MAIN MONITOR (CPU+DRAM)
// ============================================================

class RaplMonitor {
public:
    RaplDomain pkg;   // CPU package (J)
    RaplDomain dram;  // DRAM domain (J)

    double startTime = 0;  // s
    double lastTime  = 0;  // s
    double totalTime = 0;  // s

    RaplMonitor()
        : pkg("/sys/class/powercap/intel-rapl:0"),
          dram("/sys/class/powercap/intel-rapl:0:0")
    {
        startTime = now();
        lastTime = startTime;
    }

    static double now() {
        using namespace std::chrono;
        return duration<double>(
            steady_clock::now().time_since_epoch()
        ).count();
    }

    // Sample instantaneous CPU + DRAM power (W)
    void sample(double &Ppkg, double &Pdram) {
        double t = now();
        double dt = t - lastTime;
        if (dt <= 0) dt = 1e-9;

        double dE_pkg = 0, dE_dram = 0;
        pkg.sample(dE_pkg);
        dram.sample(dE_dram);

        Ppkg  = dE_pkg  / dt;  // W = J/s
        Pdram = dE_dram / dt;  // W = J/s

        totalTime = t - startTime;
        lastTime = t;
    }

    // Average power over lifetime (W)
    double cpu_avg_power()  const { return pkg.totalEnergyJ  / totalTime; }
    double dram_avg_power() const { return dram.totalEnergyJ / totalTime; }

    // Total energy (J)
    double cpu_energy()  const { return pkg.totalEnergyJ; }
    double dram_energy() const { return dram.totalEnergyJ; }
};


// ============================================================
//       GLOBALS + THREAD + PUBLIC API (with units)
// ============================================================

static pthread_t CPUPowerThread;
static std::atomic<bool> CPUpollThreadStatus{false};
static std::string CPUfilename;
static int CPU_SAMPLE_MS = 50;

static RaplMonitor* rapl = nullptr;

static void* CPUPowerPollingFunc(void*)
{
    FILE* fp = fopen(CPUfilename.c_str(), "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open file %s\n", CPUfilename.c_str());
        return nullptr;
    }

    // Column header with units
    fprintf(fp,
        "# timestep  CPU_power(W)  DRAM_power(W)  "
        "CPU_energy(J)  DRAM_energy(J)  "
        "avgCPU(W)  avgDRAM(W)  time(s)\n"
    );

    int timestep = 0;

    while (CPUpollThreadStatus.load()) {
        usleep(1000 * CPU_SAMPLE_MS);
        timestep++;

        double Ppkg, Pdram; // W
        rapl->sample(Ppkg, Pdram);

        double Epkg = rapl->cpu_energy();     // J
        double Ed   = rapl->dram_energy();    // J
        double Apkg = rapl->cpu_avg_power();  // W
        double Ad   = rapl->dram_avg_power(); // W
        double T    = rapl->totalTime;        // s

        // Log to file
        fprintf(fp, "%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
                timestep, Ppkg, Pdram, Epkg, Ed, Apkg, Ad, T);
        fflush(fp);

        // Console update
        printf("\r[CPU = %.5f W] [DRAM = %.5f W]", Ppkg, Pdram);
        fflush(stdout);
    }

    fclose(fp);
    return nullptr;
}


// ============================================================
//          PUBLIC BEGIN/END API WITH UNITS IN OUTPUT
// ============================================================

inline void CPUPowerBegin(const char* alg, int ms=50)
{
    CPU_SAMPLE_MS = ms;
    CPUpollThreadStatus = true;

    CPUfilename = std::string("power.dat");
    rapl = new RaplMonitor();

    int code = pthread_create(&CPUPowerThread, nullptr, CPUPowerPollingFunc, nullptr);
    if (code) {
        fprintf(stderr, "pthread_create error %d\n", code);
        exit(1);
    }

    // small cooldown
    usleep(1000000);
}

inline void CPUPowerEnd()
{
    // small cooldown
    usleep(1000000);

    CPUpollThreadStatus = false;
    pthread_join(CPUPowerThread, nullptr);

    double ckWh = 3600000.0; // J → kWh

    printf("\n\n========== CPU + DRAM POWER SUMMARY ==========\n");

    printf("CPU Avg. Power:      %.6f W\n",   rapl->cpu_avg_power());
    printf("CPU Total Energy:    %.6f J (%.9f kWh)\n",
        rapl->cpu_energy(), rapl->cpu_energy()/ckWh);

    printf("DRAM Avg. Power:     %.6f W\n",   rapl->dram_avg_power());
    printf("DRAM Total Energy:   %.6f J (%.9f kWh)\n",
        rapl->dram_energy(), rapl->dram_energy()/ckWh);

    printf("Total Measurement Time: %.6f s\n", rapl->totalTime);

    printf("===============================================\n\n");

    delete rapl;
}
