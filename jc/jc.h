/*! \file jc.h
*  \brief Entry point Janzen-Connell model
*  \author Hanno Hildenbrandt
*/

#ifndef JC_H_INCLUDED
#define JC_H_INCLUDED

#include <vector>
#include <utility>
#include <tuple>
#include <set>
#include <memory>
#include <string>
#include "observer.hpp"
#include "rndutils.hpp"
#include "gdm.h"
#include "jc_config.hpp"


namespace jc {

  constexpr char ReportBugs[] = "report bugs to h.hildenbrandt@rug.nl, please";


  using reng_t = rndutils::default_engine;


  struct Parameter
  {
    int L = 1000;                       ///< Area size
    int L0 = 1000;                      ///< Initial area size
    int Linc = 1;
    int Lg = 1000;                      ///< Area increment every Lg years
    double mu = 0.01;                   ///< Individual death rate
    double nu = 1e-15;                  ///< Speciation rate 
    double s_nu = 500;                  ///< sigma speciation radius
    double Psi = 0.1;                   ///< Strength of PJC effect
    double Phi = 100000.0;              ///< width of PJC effect wrt genetic distance
    double s_jc = -1.0;                 ///< width of PJC effect wrt spatial distance
    double s_disp = -1.0;               ///< width of dispersal kernal
    int jc_cutoff = 0;
    int disp_cutoff = 0;
    int64_t ticks = 10000000;           ///< number of turnovers
    std::string filename = "";          ///< file name result file
    std::string cont = "";              ///< continuation file
    int64_t log = 0;                    ///< log interval for M, D and R [events]
    double elapsed_time = 0.0;
    bool verbose = false;               ///< verbose cmd line output flag
    bool profile = false;
    int64_t totalTicks = 0;             ///< total ticks
    size_t maxMB = 64 * 1024;           ///< max lookup table size [MB]
    std::string batch;                  ///< parameter file
    bool torus = false;
    int nthreads = 1;
    int simId = 0;
    int checkInterval = 0;              ///< basic checks every checkInterval seconds, see CheckObs
    bool summary = false;
    int64_t max_seq_num = max_sequence_number;
  };


  // Event record
  struct event
  {
    int64_t T;                // time
    std::size_t NS;           // number of species after the event
    position pos;             // position on grid
    int sp;                   // affected species
    int ancestor;             // ancestor
  };


  class Kernel
  {
    Kernel(square_buffer<real_t>&& K) :
      K_(std::move(K)),
      cdf_(K_.cbegin(), K_.cend())
    {
    }

  public:
    friend Kernel make_dispersal_kernel(const Parameter& param);
    friend Kernel make_spar_kernel(const Parameter& param);

    Kernel() {}

    const auto& K() const noexcept { return K_; }

    auto sum() const noexcept { return cdf_.cdf().back(); }

    template <typename RENG>
    int cdf(RENG& reng) const { return cdf_(reng); }

  private:
    const square_buffer<real_t> K_;
    const rndutils::mutable_discrete_distribution<int> cdf_;
  };


  // model kernels
  struct kernels_t
  {
    explicit kernels_t(const Parameter& param);

    const Kernel Kdisp;
    const Kernel Kjc;
  };


  class Model
  {
  public:
    explicit Model(const Parameter& param);
    ~Model() {}
    int64_t run(class Observer* obs);

    // Observer access
    const Parameter& param() const { return param_; }
    const square_buffer<int>& M() const { return M_; }              // Grid
    const std::vector<int>& R() const { return R_; }                // Abundance vector
    const kernels_t& kernels() const { return kernels_; }           // spatial kernels
    const GDM& gdm() const { return *gdm_; }                        // Genetic distance matrix
    const std::vector<event>& events() const { return events_; };   // Event log

  private:
    // simulate one timestep
    // returns {number extinction, number speciations, number of death events}
    tick_stats timestep(int64_t T);
    void area_growth(int64_t T);
    void generate_random_positions();
    void shift_random_positions();
    void convert_R();

    Parameter param_;
    rndutils::default_engine reng_;
    int64_t T0_;                      // start time step
    square_buffer<int> M_;            // Grid
    std::vector<int> R_;              // Abundance vector
    std::vector<real_t> Rf_;          // Abundance vector converted to floating point
    kernels_t kernels_;               // constant kernels
    std::unique_ptr<GDM> gdm_;        // Genetic distance matrix

    struct sampler_event { position pos; int sp = 0; int rsp = 0; };
    std::vector<sampler_event, tbb::cache_aligned_allocator<sampler_event>> sampler_events_;
    std::set<position> sampler_pos_;
    std::vector<sampler_event, tbb::cache_aligned_allocator<sampler_event>> area_growth_events_;
    std::vector<event> events_;                       // Event log
    std::vector<int> rtest_;
    struct tls_guard { explicit tls_guard(const Parameter&) noexcept; } tls_guard_;
  };

}


#endif
