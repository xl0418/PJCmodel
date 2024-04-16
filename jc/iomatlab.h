/*! \file iomatlab.h
*  \brief input-output from Matlab m-files
*  \author Hanno Hildenbrandt
*/

#ifndef JANZECONNELL_IOMATLAB_H_INCLUDED
#define JANZECONNELL_IOMATLAB_H_INCLUDED

#include <fstream>
#include <utility>
#include "rndutils.hpp"
#include "jc.h"
#include "square_buffer.h"
#include "gdm.h"


namespace jc {


  // to facilitate parsing, these labels are inserted into the result stream.
  const char cpp_skip_label_read_back[] = "% cpp_skip_label_read_back";
  const char cpp_skip_label_last_log[] = "% cpp_skip_label_last_log";
  const char cpp_skip_label_event_log[] = "% cpp_skip_label_event_log";
  const char cpp_skip_label_state[] = "cpp_skip_label_state";
  const char cpp_skip_label_m[] = "% cpp_skip_label_m";


  struct Continuation
  {
    explicit Continuation(const std::string& fname);

    std::vector<int> M_;        // Grid
    std::vector<int> R_;        // Abundance vector
    std::vector<int> D_;    // Genetic distance matrix    
    Parameter param_;
  };


  std::istream& skip_to_label(std::istream& is, const char* label);
  std::ostream& operator<<(std::ostream& os, const event& e);
  std::istream& operator>>(std::istream& is, event& e);
  std::istream& operator>>(std::istream& is, Parameter& p);


  class MatLogger
  {
  public:
    MatLogger(const Parameter& param);
    ~MatLogger();

    void logEvents(std::vector<event> const& events);

    void logSnapshot(int64_t T,
                     const GDM& D,
                     std::vector<int> const& R);

    void logLastSnapshot(int64_t T, square_buffer<int> const& M,
                         const GDM& D,
                         std::vector<int> const& R);

    void logState(const Parameter& param);

  private:
    std::ofstream os_;
  };


}

#endif
