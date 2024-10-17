/* -*- c++ -*- ----------------------------------------------------------
     PWmat-MLFF to LAMMPS
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(mlff, PairMlff);
// clang-format on
#else



#ifndef LMP_PAIR_MLFF_H
#define LMP_PAIR_MLFF_H

#include "pair.h"
#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>


namespace LAMMPS_NS {

    class PairMlff : public Pair {
        public:
            PairMlff(class LAMMPS *);
            ~PairMlff() override;

            void compute(int, int) override;
            void settings(int, char **) override;
            void coeff(int, char **) override;
            double init_one(int, int) override;
            void init_style() override;
            int get_node_rank();
            void calculate_neighbor();

        protected:
            virtual void allocate();
        
        private:
            torch::jit::script::Module model;
            torch::Device device = torch::kCPU;
            torch::Dtype dtype = torch::kFloat32;
            std::vector<int> type_map;
            std::vector<int> image_type;
            std::vector<int> neighbor_type;
            std::vector<int> neighbor_list;
            std::vector<double> image_dR;
      
            double cutoff;
            int max_neighbor = 0;
            std::vector<int> neighbor_map = {};
            std::vector<int> neighbor_width = {0};
    };

}
#endif
#endif
