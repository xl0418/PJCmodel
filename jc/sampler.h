#ifndef JANZENCONNELL_SAMPLER_H_INCLUDED
#define JANZENCONNELL_SAMPLER_H_INCLUDED

#include <vector>
#include <utility>
#include "jc.h"
#include "gdm.h"


namespace jc { 
  

  class Sampler
  {
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

  protected:
    Sampler() {}

  public:
    // model/shared sampler state
    struct state_t
    {
      const square_buffer<int>& M;        // Grid
      const GDM& D;                       // Genetic distance matrix
      const std::vector<real_t>& Rf;      // Abundance vector converted to floating point
    };


    static Sampler* create(const Parameter& param);
    static const char* get_name(const Parameter& params);

    virtual ~Sampler() {}

    virtual int draw(const position& pos,
                     const state_t& state,
                     const kernels_t& kernels,   
                     reng_t& reng) = 0;
  };


} // namespace jc

#endif
