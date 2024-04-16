/*! \file gdm.h
 *  \brief Genetic distance matrix
 *  \author Hanno Hildenbrandt
 */

#ifndef JC_GDM_H_INCLUDED
#define JC_GDM_H_INCLUDED

#include <cstdint>
#include <vector>
#include <utility>
#include "rndutils.hpp"
#include "square_buffer.h"
#include "jc_config.hpp"


namespace jc { 


  class GDM
  {
  public:
    // Creates the initial 2x2 matrix.
    // 
    // Initialize the GDM to:
    //   0 1
    //   1 0
    explicit GDM(const struct Parameter& param);

    GDM(const struct Parameter& param, const std::vector<real_t>& Rf, square_buffer<int>&& Dij);

    void extinction(int speciesId);                // Handle extinction
    int specification(const struct position& pos, int nsp, square_buffer<int>& M);       // Handle specification

    // increase off-diagonal elements by incr
    // re-calculates expdij(Dij), Pimp_ and PimpCdf_
    void update(int L, const std::vector<real_t>& Rf, int incr);

    size_t n() const noexcept { return Dij_.n(); }
    const square_buffer<int>& Dij() const noexcept { return Dij_; }
    const square_buffer<real_t>& expDij() const noexcept { return expDij_; }

    // implicit probabilities
    const auto& Pimp() const noexcept { return Pimp_; }
    const auto& PimpCDF() const noexcept { return PimpCDF_; }

  private:
    void update_simd(const std::vector<real_t>& Rf, int incr);

    square_buffer<int> Dij_;        // holds Dij, the genetic distance matrix
    square_buffer<real_t> expDij_;  // holds expdij(Dij)
    std::vector<real_t> Pimp_;      // holds implicit probability per species
    rndutils::mutable_discrete_distribution<int, rndutils::all_zero_policy_uni> PimpCDF_;
    const std::vector<real_t> expdij_lookup_;
    rndutils::default_engine reng_;
    const double s_nu_;   // speciation sigma
    const double Psi_;    // Pimp scaling factor
  };


} // namespace jc

#endif
