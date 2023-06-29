#pragma once

#include <vector>

#include "strings/stringcontainer.hpp"
#include "merge/bingmann-lcp_losertree.hpp"
#include "mpi/environment.hpp"


inline size_t pow2roundup(size_t x) {
  if (x == 0u)
    return 1u;
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

template<typename AllToAllStringPolicy, size_t K, typename StringSet>
    static inline dss_schimek::StringLcpContainer<StringSet> merge(
        dss_schimek::StringLcpContainer<StringSet>&& recv_string_cont,
        std::vector<std::pair<size_t, size_t>>& ranges,
        const size_t num_recv_elems) {
      
      dss_schimek::mpi::environment env;

      if (recv_string_cont.size() == 0u)
        return dss_schimek::StringLcpContainer<StringSet>();

      const size_t nextPow2 = pow2roundup(env.size());
      for (size_t i = env.size(); i < nextPow2; ++i)
        ranges.emplace_back(0u, 0u);
      

      std::vector<typename StringSet::String> sorted_string(recv_string_cont.size());
      std::vector<size_t> sorted_lcp(recv_string_cont.size());
      StringSet ss = recv_string_cont.make_string_set();

      dss_schimek::StringLcpPtrMergeAdapter<StringSet> mergeAdapter(ss, recv_string_cont.lcp_array());
      dss_schimek::LcpStringLoserTree_<K, StringSet> loser_tree(mergeAdapter, ranges.data());
      StringSet sortedSet(sorted_string.data(), sorted_string.data() + sorted_string.size());
      dss_schimek::StringLcpPtrMergeAdapter out_(sortedSet, sorted_lcp.data());

      std::vector<size_t> oldLcps;
      if (AllToAllStringPolicy::PrefixCompression) {
        loser_tree.writeElementsToStream(out_, num_recv_elems, oldLcps);
      }
      else {
        loser_tree.writeElementsToStream(out_, num_recv_elems);
      }
      dss_schimek::StringLcpContainer<StringSet> sorted_string_cont;//(std::move(recv_string_cont));

      sorted_string_cont.set(std::move(recv_string_cont.raw_strings()));
      sorted_string_cont.set(std::move(sorted_string));
      sorted_string_cont.set(std::move(sorted_lcp));
      sorted_string_cont.setSavedLcps(std::move(oldLcps));

      return sorted_string_cont;
    }


  template<typename AllToAllStringPolicy, typename StringLcpContainer>
    static inline StringLcpContainer choose_merge(StringLcpContainer&& recv_string_cont,
        std::vector<std::pair<size_t, size_t>> ranges,
        size_t num_recv_elems,
        dss_schimek::mpi::environment env = dss_schimek::mpi::environment()) {

      const size_t nextPow2 = pow2roundup(env.size());
      switch (nextPow2) {
        case 1 :  return merge<AllToAllStringPolicy,1>(std::move(recv_string_cont),
                      ranges,
                      num_recv_elems);
        case 2 : return merge<AllToAllStringPolicy,2>(std::move(recv_string_cont),
                     ranges,
                     num_recv_elems);
        case 4 : return merge<AllToAllStringPolicy,4>(std::move(recv_string_cont),
                     ranges,
                     num_recv_elems);
        case 8 : return merge<AllToAllStringPolicy,8>(std::move(recv_string_cont),
                     ranges,
                     num_recv_elems);
        case 16 : return merge<AllToAllStringPolicy,16>(std::move(recv_string_cont),
                      ranges,
                      num_recv_elems);
        case 32 : return merge<AllToAllStringPolicy,32>(std::move(recv_string_cont),
                      ranges,
                      num_recv_elems);
        case 64 : return merge<AllToAllStringPolicy,64>(std::move(recv_string_cont),
                      ranges,
                      num_recv_elems);
        case 128 : return merge<AllToAllStringPolicy,128>(std::move(recv_string_cont),
                       ranges,
                       num_recv_elems);
        case 256 : return merge<AllToAllStringPolicy,256>(std::move(recv_string_cont),
                       ranges,
                       num_recv_elems);
        case 512 : return merge<AllToAllStringPolicy,512>(std::move(recv_string_cont),
                       ranges,
                       num_recv_elems);
        case 1024 : return merge<AllToAllStringPolicy,1024>(std::move(recv_string_cont),
                       ranges,
                       num_recv_elems);
        case 2048: return merge<AllToAllStringPolicy,2048>(std::move(recv_string_cont),
                       ranges,
                       num_recv_elems);
        case 4096 : return merge<AllToAllStringPolicy,4096>(std::move(recv_string_cont),
                       ranges,
                       num_recv_elems);
        case 8192 : return merge<AllToAllStringPolicy,8192>(std::move(recv_string_cont),
                       ranges,
                       num_recv_elems);
        case 16384 : return merge<AllToAllStringPolicy,16384>(std::move(recv_string_cont),
                       ranges,
                       num_recv_elems);
        default : std::cout << "Error in merge: K is not 2^i for i in {0,...,14} " << std::endl; 
                  std::abort();
      }
      return StringLcpContainer();
    }
