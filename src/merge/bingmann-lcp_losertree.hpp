/*******************************************************************************
 * src/sequential/bingmann-lcp_losertree.hpp
 *
 * Implementation of a LCP aware multiway losertree.
 *
 *******************************************************************************
 * Copyright (C) 2013-2014 Andreas Eberle <email@andreas-eberle.com>
 * Copyright (C) 2014 Timo Bingmann <tb@panthema.net>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <iostream>
#include <utility>

#include <tlx/define/likely.hpp>

#include "merge/stringtools.hpp"
#include "strings/stringptr.hpp"

namespace bingmann {

using namespace stringtools;

typedef unsigned char* string;

/******************************************************************************/
// LcpStringLoserTree

template <size_t K>
class LcpStringLoserTree {
    typedef LcpStringPtr Stream;

    struct Node {
        size_t idx;
        lcp_t lcp;
    };

private:
    Stream streams[K + 1];
    Node nodes[K + 1];

    //! play one comparison edge game: contender is the node below
    //! defender. After the game, defender contains the lower index, contender
    //! the winning index, and defender.lcp = lcp(s_loser,s_winner).
    void updateNode(Node& contender, Node& defender) {
        Stream const& defenderStream = streams[defender.idx];

        if (TLX_UNLIKELY(defenderStream.empty()))
            return;

        Stream const& contenderStream = streams[contender.idx];

        if (TLX_UNLIKELY(contenderStream.empty())) {
            std::swap(defender, contender);
            return;
        }
#if 1
        if (defender.lcp > contender.lcp) {
            // CASE 2: curr->lcp > contender->lcp => curr < contender
            std::swap(defender, contender);
        } else if (defender.lcp == contender.lcp) {
            // CASE 1: compare more characters
            lcp_t lcp = defender.lcp;

            string s1 = defenderStream.firstString() + lcp;
            string s2 = contenderStream.firstString() + lcp;

            // check the strings starting after lcp and calculate new lcp
            while (*s1 != 0 && *s1 == *s2)
                s1++, s2++, lcp++;

            if (*s1 < *s2) // CASE 1.1: curr < contender
                std::swap(defender, contender);

            // update inner node with lcp(s_1,s_2)
            defender.lcp = lcp;
        } else {
            // CASE 3: curr->lcp < contender->lcp => contender < curr  => nothing to do
        }
#else
        lcp_compare(
            contender.idx,
            contenderStream.firstString(),
            contender.lcp,
            defender.idx,
            defenderStream.firstString(),
            defender.lcp,
            contender.idx,
            contender.lcp,
            defender.idx,
            defender.lcp
        );
#endif
        assert(
            scmp(streams[contender.idx].firstString(), streams[defender.idx].firstString()) <= 0
        );

        assert(
            calc_lcp(streams[contender.idx].firstString(), streams[defender.idx].firstString())
            == defender.lcp
        );
    }

    void initTree(lcp_t knownCommonLcp) {
        // std::cout << "inittree start\n";
        for (size_t k = 1; k <= K; k++) {
            Node contender;
            contender.idx = k;
            contender.lcp = knownCommonLcp;

            size_t nodeIdx = K + k;

            // std::cout << "nodeIdx " << nodeIdx << "\n";

            while (nodeIdx % 2 == 0 && nodeIdx > 2) {
                nodeIdx >>= 1;
                // std::cout << "play against " << nodeIdx << "\n";
                updateNode(contender, nodes[nodeIdx]);
            }
            nodeIdx = (nodeIdx + 1) / 2;
            // std::cout << "save as " << nodeIdx << "\n";
            nodes[nodeIdx] = contender;
        }
        // std::cout << "inittree done\n";
    }

public:
    LcpStringLoserTree(
        LcpStringPtr const& input,
        std::vector<size_t> const& offsets,
        std::vector<size_t> const& sizes,
        lcp_t knownCommonLcp = 0
    ) {
        assert(sizes.size() == K && offsets.size() == K);
        auto op = [&](auto const& offset, auto const& size) { return input.sub(offset, size); };
        std::transform(offsets.begin(), offsets.end(), sizes.begin(), streams + 1, op);

        initTree(knownCommonLcp);
    }

    void writeElementsToStream(LcpStringPtr outStream, const size_t length) {
        const LcpStringPtr end = outStream.sub(length, 0);
        while (outStream < end) {
            // take winner and put into output

            size_t winnerIdx = nodes[1].idx;
            // std::cout << "winnerIdx " << winnerIdx << std::endl;

            outStream.setFirst(streams[winnerIdx].firstString(), nodes[1].lcp);
            ++outStream;

            // advance winner stream

            Stream& stream = streams[winnerIdx];
            ++stream;

            // run new items from winner stream up the tree

            Node& contender = nodes[1];

            if (!stream.empty())
                contender.lcp = streams[winnerIdx].firstLcp();

            size_t nodeIdx = winnerIdx + K;
            // std::cout << "nodeIdx " << nodeIdx << "\n";

            while (nodeIdx > 2) {
                nodeIdx = (nodeIdx + 1) / 2;
                // std::cout << "play against " << nodeIdx << "\n";
                updateNode(contender, nodes[nodeIdx]);
            }
            // std::cout << "play against " << nodeIdx << "\n";

            // for (size_t nodeIdx = (K + winnerIdx) >> 1; nodeIdx >= 1; nodeIdx >>= 1)
            // {
            //     updateNode(contender, nodes[nodeIdx]);
            // }
        }
    }
};

} // namespace bingmann

namespace dss_schimek {
template <size_t K, typename StringSet>
class LcpStringLoserTree_ {
    using Stream = dss_schimek::StringLcpPtrMergeAdapter<StringSet>;
    using CharIt = typename StringSet::CharIterator;

    struct Node {
        size_t idx;
        lcp_t lcp;
    };

private:
    Stream streams[K + 1];
    Node nodes[K + 1];

    //! play one comparison edge game: contender is the node below
    //! defender. After the game, defender contains the lower index, contender
    //! the winning index, and defender.lcp = lcp(s_loser,s_winner).
    void updateNodeCompressedPrefix(Node& contender, Node& defender) {
        Stream const& defenderStream = streams[defender.idx];

        if (TLX_UNLIKELY(defenderStream.empty()))
            return;

        Stream const& contenderStream = streams[contender.idx];

        if (TLX_UNLIKELY(contenderStream.empty())) {
            std::swap(defender, contender);
            return;
        }
#if 1
        if (defender.lcp > contender.lcp) {
            // CASE 2: curr->lcp > contender->lcp => curr < contender
            std::swap(defender, contender);
        } else if (defender.lcp == contender.lcp) {
            // CASE 1: compare more characters
            lcp_t lcp = defender.lcp;

            CharIt s1 = defenderStream.firstStringChars() + (lcp - defenderStream.firstLcp());
            CharIt s2 = contenderStream.firstStringChars() + (lcp - contenderStream.firstLcp());
            // std::cout << "\t\tdefender: " << s1 << std::endl;
            // std::cout << "\t\tcontender: " << s2 << std::endl;

            // check the strings starting after lcp and calculate new lcp
            while (*s1 != 0 && *s1 == *s2)
                s1++, s2++, lcp++;

            if (*s1 < *s2) // CASE 1.1: curr < contender
                std::swap(defender, contender);

            // update inner node with lcp(s_1,s_2)
            defender.lcp = lcp;
        } else {
            // CASE 3: curr->lcp < contender->lcp => contender < curr  => nothing to do
        }
#else
        lcp_compare(
            contender.idx,
            contenderStream.firstString(),
            contender.lcp,
            defender.idx,
            defenderStream.firstString(),
            defender.lcp,
            contender.idx,
            contender.lcp,
            defender.idx,
            defender.lcp
        );
#endif
        // todo is this postcondition correct?
        // assert(
        //     scmp(
        //         streams[contender.idx].firstStringChars(),
        //         streams[defender.idx].firstStringChars()
        //     )
        //     <= 0
        // );
        //
        // assert(
        //     calc_lcp(
        //         streams[contender.idx].firstStringChars(),
        //         streams[defender.idx].firstStringChars()
        //     )
        //     == defender.lcp
        // );
    }

    void updateNode(Node& contender, Node& defender) {
        Stream const& defenderStream = streams[defender.idx];

        if (TLX_UNLIKELY(defenderStream.empty()))
            return;

        Stream const& contenderStream = streams[contender.idx];

        if (TLX_UNLIKELY(contenderStream.empty())) {
            std::swap(defender, contender);
            return;
        }
#if 1
        if (defender.lcp > contender.lcp) {
            // CASE 2: curr->lcp > contender->lcp => curr < contender
            std::swap(defender, contender);
        } else if (defender.lcp == contender.lcp) {
            // CASE 1: compare more characters
            lcp_t lcp = defender.lcp;

            CharIt s1 = defenderStream.firstStringChars() + lcp;
            CharIt s2 = contenderStream.firstStringChars() + lcp;
            // std::cout << "\t\tdefender: " << s1 << std::endl;
            // std::cout << "\t\tcontender: " << s2 << std::endl;

            // check the strings starting after lcp and calculate new lcp
            while (*s1 != 0 && *s1 == *s2)
                s1++, s2++, lcp++;

            if (*s1 < *s2) // CASE 1.1: curr < contender
                std::swap(defender, contender);

            // update inner node with lcp(s_1,s_2)
            defender.lcp = lcp;
        } else {
            // CASE 3: curr->lcp < contender->lcp => contender < curr  => nothing to do
        }
#else
        lcp_compare(
            contender.idx,
            contenderStream.firstString(),
            contender.lcp,
            defender.idx,
            defenderStream.firstString(),
            defender.lcp,
            contender.idx,
            contender.lcp,
            defender.idx,
            defender.lcp
        );
#endif
        assert(
            scmp(
                streams[contender.idx].firstStringChars(),
                streams[defender.idx].firstStringChars()
            )
            <= 0
        );

        assert(
            calc_lcp(
                streams[contender.idx].firstStringChars(),
                streams[defender.idx].firstStringChars()
            )
            == defender.lcp
        );
    }

    void initTree(lcp_t knownCommonLcp) {
        // std::cout << "inittree start\n";
        for (size_t k = 1; k <= K; k++) {
            Node contender;
            contender.idx = k;
            contender.lcp = knownCommonLcp;

            size_t nodeIdx = K + k;

            // std::cout << "nodeIdx " << nodeIdx << "\n";

            while (nodeIdx % 2 == 0 && nodeIdx > 2) {
                nodeIdx >>= 1;
                // std::cout << "play against " << nodeIdx << "\n";
                updateNode(contender, nodes[nodeIdx]);
            }
            nodeIdx = (nodeIdx + 1) / 2;
            // std::cout << "save as " << nodeIdx << "\n";
            nodes[nodeIdx] = contender;
        }
        // std::cout << "inittree done\n";
    }

public:
    LcpStringLoserTree_(
        dss_schimek::StringLcpPtrMergeAdapter<StringSet> const& input,
        std::vector<size_t> const& offsets,
        std::vector<size_t> const& sizes,
        lcp_t knownCommonLcp = 0
    ) {
        assert(sizes.size() == K && offsets.size() == K);
        auto op = [&](auto const& offset, auto const& size) { return input.sub(offset, size); };
        std::transform(offsets.begin(), offsets.end(), sizes.begin(), streams + 1, op);
        initTree(knownCommonLcp);
    }

    void writeElementsToStream(
        dss_schimek::StringLcpPtrMergeAdapter<StringSet> outStream,
        const size_t length,
        std::vector<size_t>& oldLcps
    ) {
        const dss_schimek::StringLcpPtrMergeAdapter<StringSet> end = outStream.sub(length, 0);
        oldLcps.clear();
        oldLcps.reserve(length);
        while (outStream < end) {
            // take winner and put into output
            // std::cout << "after tournament" << std::endl;
            // for (size_t i = 1; i < K + 1; ++i) {
            //  std::cout << i << " idx: " << nodes[i].idx << " string: " <<
            //  streams[nodes[i].idx].firstStringChars() << " lcp: " << nodes[i].lcp << std::endl;
            //}
            // std::cout << "\n\n";

            size_t winnerIdx = nodes[1].idx;

            // outStream.setFirst(streams[winnerIdx].firstString(), nodes[1].lcp);
            // std::cout << streams[winnerIdx].firstStringChars() << " lcp: " <<
            // streams[winnerIdx].firstLcp() << std::endl;
            outStream.setFirst(streams[winnerIdx].firstString(), nodes[1].lcp);
            oldLcps.emplace_back(streams[winnerIdx].firstLcp());

            ++outStream;

            // advance winner stream

            Stream& stream = streams[winnerIdx];
            ++stream;
            // run new items from winner stream up the tree

            Node& contender = nodes[1];

            if (!stream.empty())
                contender.lcp = streams[winnerIdx].firstLcp();

            size_t nodeIdx = winnerIdx + K;
            // std::cout << "nodeIdx " << nodeIdx << "\n";
            // std::cout << "before tournament" << std::endl;
            // for (size_t i = 1; i < K + 1; ++i) {
            //   std::cout << i << " idx: " << nodes[i].idx << " string: " <<
            //   streams[nodes[i].idx].firstStringChars() << " lcp: " << nodes[i].lcp << std::endl;
            // }
            // std::cout << "\n\n";
            while (nodeIdx > 2) {
                nodeIdx = (nodeIdx + 1) / 2;
                // std::cout << "play against " << nodeIdx << "\n";
                updateNodeCompressedPrefix(contender, nodes[nodeIdx]);
            }
            // std::cout << "play against " << nodeIdx << "\n";

            // for (size_t nodeIdx = (K + winnerIdx) >> 1; nodeIdx >= 1; nodeIdx >>= 1)
            // {
            //     updateNode(contender, nodes[nodeIdx]);
            // }
        }
    }
    void writeElementsToStream(
        dss_schimek::StringLcpPtrMergeAdapter<StringSet> outStream, const size_t length
    ) {
        const dss_schimek::StringLcpPtrMergeAdapter<StringSet> end = outStream.sub(length, 0);
        while (outStream < end) {
            // take winner and put into output
            // std::cout << "after tournament" << std::endl;
            // for (size_t i = 1; i < K + 1; ++i) {
            //  std::cout << i << " idx: " << nodes[i].idx << " string: " <<
            //  streams[nodes[i].idx].firstStringChars() << " lcp: " << nodes[i].lcp << std::endl;
            //}
            // std::cout << "\n\n";

            size_t winnerIdx = nodes[1].idx;

            // outStream.setFirst(streams[winnerIdx].firstString(), nodes[1].lcp);
            // std::cout << streams[winnerIdx].firstStringChars() << " lcp: " <<
            // streams[winnerIdx].firstLcp() << std::endl;
            outStream.setFirst(streams[winnerIdx].firstString(), nodes[1].lcp);

            ++outStream;

            // advance winner stream

            Stream& stream = streams[winnerIdx];
            ++stream;
            // run new items from winner stream up the tree

            Node& contender = nodes[1];

            if (!stream.empty())
                contender.lcp = streams[winnerIdx].firstLcp();

            size_t nodeIdx = winnerIdx + K;
            // std::cout << "nodeIdx " << nodeIdx << "\n";
            // std::cout << "before tournament" << std::endl;
            // for (size_t i = 1; i < K + 1; ++i) {
            //   std::cout << i << " idx: " << nodes[i].idx << " string: " <<
            //   streams[nodes[i].idx].firstStringChars() << " lcp: " << nodes[i].lcp << std::endl;
            // }
            // std::cout << "\n\n";
            while (nodeIdx > 2) {
                nodeIdx = (nodeIdx + 1) / 2;
                // std::cout << "play against " << nodeIdx << "\n";
                updateNode(contender, nodes[nodeIdx]);
            }
            // std::cout << "play against " << nodeIdx << "\n";

            // for (size_t nodeIdx = (K + winnerIdx) >> 1; nodeIdx >= 1; nodeIdx >>= 1)
            // {
            //     updateNode(contender, nodes[nodeIdx]);
            // }
        }
    }
};
} // namespace dss_schimek

/******************************************************************************/
