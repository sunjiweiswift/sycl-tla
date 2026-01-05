/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "fmha_fusion.hpp"

namespace cutlass::fmha {

template <int Stages> class XeDefault {};   // Default FMHA mainloop, P in registers.

};

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy_,
          bool CausalMask_,
          class TiledMMAQK_,          // Tiling for Q*K GEMM
          class TiledMMAPV_,          // Tiling for P*V GEMM
          int VTiles_,                // # of tiles in V dimension
          class TensorQ_,             // Global Q/K/V tensors
          class TensorK_,
          class TensorV_,
          class TiledCopyQ_ = void,   // Optional TiledCopy for loading Q
          class TiledCopyK_ = void,   // Optional TiledCopy for loading K
          class TiledCopyV_ = void>   // Optional TiledCopy for loading V
struct FMHAFwdMainloop {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages,
          bool CausalMask_,
          class TiledMMAQK_, class TiledMMAPV_, int VTiles_,
          class TensorQ_, class TensorK_, class TensorV_,
          class TiledCopyQ_, class TiledCopyK_, class TiledCopyV_>
struct FMHAFwdMainloop<XeDefault<Stages>, CausalMask_,
                       TiledMMAQK_, TiledMMAPV_, VTiles_,
                       TensorQ_, TensorK_, TensorV_,
                       TiledCopyQ_, TiledCopyK_, TiledCopyV_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  static constexpr int HeadDim = VTiles * get<1>(TileShapePV{});
  static constexpr int HeadDimTiles = HeadDim / get<2>(TileShapeQK{});
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(take<1,4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));
  static constexpr int QGroupSize = 1;

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using TensorQ2D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_,_),0)));
  using TensorQ3D = decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_,_,_),0)));
  using TensorK2D = decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_,_),0)));
  using TensorV2D = decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_,_),0)));


  using TiledCopyQ = conditional_t<is_void_v<TiledCopyQ_>, decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})), TiledCopyQ_>;
  using TiledCopyK = conditional_t<is_void_v<TiledCopyK_>, decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})), TiledCopyK_>;
  using TiledCopyV = conditional_t<is_void_v<TiledCopyV_>, decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})), TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
                           make_identity_tensor(select<0,1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using ElementQ = typename TiledMMAQK::ValTypeA;
  using ElementK = typename TiledMMAQK::ValTypeB;
  using ElementS = typename TiledMMAQK::ValTypeD;

  using PackTypeQ = uint128_t;
  static constexpr int PackQRatio = sizeof(PackTypeQ) / sizeof(ElementQ);
  static constexpr int QElemsPerThr = (get<0>(TileShapeQK{}) * get<2>(TileShapeQK{})) /
                                  (SGPerWG{} * intel::sg_size);
  using StoreTrait = Copy_Traits<XE_1D_STSM<
      PackTypeQ,
      typename uint_bit<sizeof_bits_v<ElementQ>>::type>>;
  using LoadTrait = Copy_Traits<XE_1D_LDSM<
      typename uint_bit<sizeof_bits_v<ElementQ>>::type,
      PackTypeQ>>;
  using TiledStoreQSmem = decltype(make_tiled_copy(
      Copy_Atom<StoreTrait, ElementQ>{},
      Layout<Shape<intel::_SGSize, SGPerWG>, Stride<_1, intel::_SGSize>>{},
      Layout<Shape<Int<PackQRatio>, Int<QElemsPerThr / PackQRatio>>, Stride<_1, Int<PackQRatio>>>{}));
  using TiledLoadQSmem = decltype(make_tiled_copy(
      Copy_Atom<LoadTrait, ElementQ>{},
      Layout<Shape<intel::_SGSize, SGPerWG>, Stride<_1, intel::_SGSize>>{},
      Layout<Shape<Int<PackQRatio>, Int<QElemsPerThr / PackQRatio>>, Stride<_1, Int<PackQRatio>>>{}));

  using SingleFragA = FragC<TiledMMAPV>;                          // (atom val,q',v')
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;     // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool CausalMask = CausalMask_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {
    // Allocate shared memory for QxK tile
    cute::array<ElementQ, QGroupSize * get<0>(TileShapeQK{}) * HeadDim> q_slm; // Assum head_size_qk == head_size_vo
    // cute::array<ElementK, get<1>(TileShapeQK{}) * HeadDim> k_preload;              // Assum head_size_qk == head_size_vo

    cute::array<ElementS, QGroupSize * get<0>(TileShapeQK{}) * HeadDim> s_slm;
    // cute::array<ElementS, QGroupSize * get<0>(TileShapeQK{})> sum_slm;
    // cute::array<ElementS, QGroupSize * get<0>(TileShapeQK{})> max_slm;

  };
private:
  SharedStorage &shared;

public:
  Params params;
  //
  // Methods
  //

  FMHAFwdMainloop(Params const &params_, SharedStorage &shared_) : params(params_), shared(shared_) {}

  static constexpr
  Params to_underlying_arguments(Arguments const &args, void * /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;            // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{val};
  }

  CUTLASS_HOST_DEVICE static
  bool can_implement(Arguments const&) {
    return true;
  }

  template <typename FragASLM, typename QVCoord>
  CUTLASS_DEVICE
  void
  operator()(TensorQ3D const& Q_3D,     // (q,d)
             TensorK2D const& K_2D,     // (k,d)
             TensorV2D const& V_2D,     // (d,k)
             FragA          & tArA,     // Output accumulator (q,v)
             FragASLM       & tArA_slm, // Output accumulator (q,v)
             FragARow       & tA_max,   // Softmax row-wise max accumulator
             FragARow       & tA_sum,   // Softmax row-wise sum accumulator
             QVCoord          blk_qv,   // WG tile indices: (Q,V)
             int              head_q,   // head index for Q
             int              blk_k0,   // K block range: [K0,K1)
             int              blk_k1,
             int              total_blk, // Total # of K blocks
             int              thr_id,
             int              seq_len,
             int              full_tile_offset,
             int              discard_seq_coord) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v = make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));
    auto tile_shape_QK = make_shape(get<0>(TileShapeQK{}), get<1>(TileShapeQK{}), C<VTiles>{} * get<2>(TileShapeQK{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(take<0,2>(Q_3D.shape()));  // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());             // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());             // (v,k)
    Tensor cP = make_identity_tensor(take<0,2>(TileShapeQK{})); // (q,k)

    /* Partition global tensors into workgroup tiles */
    Tensor gQ       = local_tile(cQ, TileShapeQK{}, append(blk_qv,_),             Step<_1,X,_1>{});   // (q,d,D)
    Tensor gK       = local_tile(cK, TileShapeQK{}, make_coord(_,_,_),            Step<X,_1,_1>{});   // (k,d,K,D)
    Tensor gV       = local_tile(cV, tile_shape_v,  make_coord(get<1>(blk_qv),_));                    // (v,k,K)
    Tensor gV_split = local_tile(gV, TileShapePV{}, make_coord(_,_,0),            Step<X,_1,_1>{});   // (v,k,VV,K)

    /* Create global -> register copies */
    
    TiledCopyQ copy_q{};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    TiledStoreQSmem store_q_smem{};
    TiledLoadQSmem load_q_smem{};
    
    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};
    
    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_store_q_smem = store_q_smem.get_slice(thr_id);
    auto thr_load_q_smem = load_q_smem.get_slice(thr_id);

    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);
    
    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);                // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);                // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);          // (atom_val,v',k',VV,K)
    
    auto sQ = make_tensor(make_smem_ptr<ElementQ>(&shared.q_slm), make_shape(get<0>(TileShapeQK{}), get<2>(TileShapeQK{}), Int<HeadDimTiles>{}, QGroupSize));
    auto tQsQStore = thr_store_q_smem.partition_D(sQ);
    auto tQsQLoad = thr_load_q_smem.partition_S(sQ);

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_,_,0));
    // auto tSrQ_fullD = thr_mma_qk.partition_sg_fragment_A(gQ_fullD(_,_,0));
    auto store_tSrQ = thr_store_q_smem.retile_S(tSrQ);
    auto load_tSrQ = thr_load_q_smem.retile_D(tSrQ);

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_,_,0,0));
    
    using SingletSrK = decltype(thr_mma_qk.partition_sg_fragment_B(gK(_,_,0,0)));   
    using SingletKrK = decltype(thr_copy_k.partition_sg_fragment_D(gK(_,_,0,0)));   
    using SingletSrK_tv_layout = decltype(SingletSrK{}.tv_layout());   
    using SingletKrK_tv_layout = decltype(SingletKrK{}.tv_layout());   
    SingletSrK_tv_layout  tSrk_tv_layout;
    SingletKrK_tv_layout  tKrk_tv_layout;
    using tSrKFull = expand_sg_fragment_t<SingletSrK, 1, HeadDimTiles>; 
    using tKrKFull = expand_sg_fragment_t<SingletKrK, 1, HeadDimTiles>; 
    tSrKFull tSrK1;
    tKrKFull tKrK1;
    
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_,_,0,0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_,_,0,0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v = make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);
              
    // ------
    // Kernel
    // ------
    auto sg = compat::get_nd_item<1>().get_sub_group();
    // int sg_id = sg.get_group_id()[0];
    auto sg_id = sg.get_group_linear_id();
    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    if (blk_k0 == 0) {
      // for (int D = 0; D < size<3>(pQgQ); D++) {
      //   prefetch(prefetch_q, pQgQ(_,_,_,D));
      // }
      // Preload Q into shared memory
      for (int q = 0; q < QGroupSize; q++) {
        TiledCopyQ copy_q{Q_3D(_,_,head_q + q)};
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < HeadDimTiles; ++D) {
          copy(copy_q, tQgQ(_,_,_,D), tQrQ);
          reorder(tQrQ, tSrQ);
          copy(store_q_smem, store_tSrQ, tQsQStore(_,_,_,D,q));
        }
      }
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      for (int D = 0; D < size<4>(pKgK); D++) {
        CUTLASS_PRAGMA_UNROLL
        for (int K = 0; K < Stages; K++) {
          prefetch(prefetch_k, pKgK(_,_,_,K,D));
        }
      }
      clear(tArA);
      fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
      clear(tA_sum);
    }

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      /* Split barrier to keep threads together */
      // barrier_arrive(ScopeWorkgroup);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      for (int q = 0; q < QGroupSize; q++) {
        /* GEMM 1: S = K * Q */
        clear(tSrS);    /* TODO: fuse w/ initial gemm call */
          copy(copy_k, tKgK(_,_,_,K,_), tKrK1);
          reorder(tKrK1, tSrK1);        
        // for (int D = 0; D < size<4>(tKgK); D++) {
        // CUTLASS_PRAGMA_UNROLL
        // for (int D = 0; D < VTiles; D++) {
        //   // copy(copy_q, tQgQ(_,_,_,D),   tQrQ);
        //   copy(copy_k, tKgK(_,_,_,K,D), tKrK);
          
        //   // reorder(tQrQ, tSrQ);
        //   // reorder(tKrK, tSrK);
        //   auto tSrK2 = make_subgroup_tensor(tSrK1(_,_,_,D), tSrk_tv_layout);
        //   reorder(tKrK, tSrK2);
        // }
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < VTiles; D++) {
        //   /* smem -> reg */
          auto tSrK = make_subgroup_tensor(tSrK1(_,_,_,D), tSrk_tv_layout);
          copy(load_q_smem, tQsQLoad(_,_,_,D,q), load_tSrQ);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }
      
        /* V prefetch for GEMM 2 */
        prefetch(prefetch_v, pVgV(_,_,_,K));
  
        /* Causal masking */
        if constexpr (CausalMask) {
        if (K == blk_k1 - 1) {
          // Need to get global col and row indices to mask the elements
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i));
            if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }
        /* k masking for remainder tiles */
        if (check_remainder_k && K == total_blk - 1) {
        FragSRow k_rem_mask;
        int k = get<0>(tKgK(0,0,0,K,0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) = (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }
  
        /* Apply softmax and scaling */
        softmax(K == 0, tSrS, tA_max, tA_sum, tArA);
        reorder(tSrS, tArP);
  
        /* GEMM 2: A += P * V, split in v dimension */
        CUTLASS_PRAGMA_UNROLL
        for (int VV = 0; VV < VTiles; VV++) {
          copy(copy_v, tVgV(_,_,_,VV,K), tVrV);
          reorder(tVrV, tArV);
          cute::gemm(mma_pv, tArP, tArV, tArA(_,_,_,VV));
        }
        // copy(copy_a, tArA_slm, tArA);
        // copy_block_r2s(tArA, tArA_slm(_,sg_id,q));
        // // copy_block_s2r(tArA, tArA_slm(_,sg_id,q));
        // if (thread(0,0)) {
        //   print("\n tArA \n");
        //   print_tensor(tArA);
        //   print("\n tArA_slm \n");
        //   print_tensor(tArA_slm(_,sg_id,q));
        // }
        /* K prefetch */
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_,_,_,K+Stages,D));
        }
      } // q loop
      // barrier_wait(ScopeWorkgroup);
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void
  softmax(bool       first_block, // First softmax block?
          FragS    & tS,          // Softmax src/dst block
          FragSRow & tS_max,      // Softmax row-wise max accumulator
          FragSRow & tS_sum,      // Softmax row-wise sum accumulator
          FragA    & tA) {        // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), params.scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(params.scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};


template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE
constexpr auto
get_sg_layout_pv(SGLayoutQK const&)
{
  return make_layout(
    get<0>(SGLayoutQK{}),
    Layout<_1, _0>{},
    get<1>(SGLayoutQK{})
  );
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <typename MMAOp,
          typename WGTileQK,
          typename SGLayoutQK,
          typename TileV>
CUTLASS_HOST_DEVICE
constexpr auto
get_tiled_mma_pv(MMAOp const&, WGTileQK const& wg_tile_qk, SGLayoutQK const& sg_layout_qk, TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(size(SGLayoutPV{}) == size(SGLayoutQK{}),
                "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

} // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
