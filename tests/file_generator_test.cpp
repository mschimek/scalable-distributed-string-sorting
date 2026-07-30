// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Tests for FileDistributer (src/util/string_generator.hpp): the file is split across the PEs by
// bytes and cut into whole lines, so concatenating the PEs' strings in rank order has to reproduce
// the file's lines. The --max-num-bytes option caps how many bytes are read; the cut is made at a
// byte boundary, so the last line may be truncated.

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <kamping/collectives/barrier.hpp>

#include "dss/mpi/communicator.hpp"
#include "dss/strings/stringcontainer.hpp"
#include "dss/strings/stringset.hpp"
#include "input/string_generator.hpp"
#include "test_util.hpp"

namespace {

using dss_test::Char;
using dss_test::Communicator;

using StringSet = dss_mehnert::StringSet<Char, dss_mehnert::Length>;
using FileGenerator = dss_mehnert::FileDistributer<StringSet>;

// `count` distinct fixed-width lines separated by '\n', with a trailing newline. Every line is
// `width` characters wide, so byte offset k * (width + 1) always lands on a line boundary.
std::string make_content(size_t const count, size_t const width) {
    std::string content;
    for (size_t i = 0; i != count; ++i) {
        auto line = std::to_string(i);
        line.insert(line.begin(), width - line.size(), '0'); // zero-pad to `width` digits
        content += line;
        content += '\n';
    }
    return content;
}

// The lines FileDistributer is expected to produce for a given byte cap: the first `max_bytes` of
// the content (all of it if `max_bytes == 0`) split at '\n', where a trailing non-empty fragment
// (a line the cut ran through) is kept as its own string. This mirrors distribute_lines exactly.
std::vector<std::string> expected_lines(std::string content, size_t const max_bytes) {
    if (max_bytes != 0 && max_bytes < content.size()) {
        content.resize(max_bytes);
    }

    std::vector<std::string> lines;
    std::string current;
    for (char const c: content) {
        if (c == '\n') {
            lines.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!current.empty()) {
        lines.push_back(current);
    }
    return lines;
}

// rank 0 writes the content to a shared path (per communicator size, so concurrent ctest runs of
// different sizes do not collide); all ranks see it after the barrier.
std::string write_shared_file(std::string const& content, Communicator const& comm) {
    auto const path = std::filesystem::temp_directory_path()
                      / ("dss_file_generator_" + std::to_string(comm.size()) + ".txt");
    if (comm.rank() == 0) {
        std::ofstream out{path, std::ios::binary | std::ios::trunc};
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    comm.barrier();
    return path.string();
}

void remove_shared_file(std::string const& path, Communicator const& comm) {
    comm.barrier(); // no rank may still be reading the file
    if (comm.rank() == 0) {
        std::filesystem::remove(path);
    }
}

// the strings of this PE, in file order, gathered across all PEs in rank order
std::vector<std::string>
read_all_lines(std::string const& path, size_t const max_bytes, Communicator const& comm) {
    FileGenerator container{path, comm, max_bytes};
    auto const ss = container.make_string_set();

    std::vector<std::string> local;
    local.reserve(ss.size());
    for (auto const& str: ss) {
        auto const* chars = ss.get_chars(str, 0);
        local.emplace_back(reinterpret_cast<char const*>(chars), ss.get_length(str));
    }
    return dss_test::gather_in_rank_order(comm, dss_test::pack(local));
}

// read `content` back through the generator with the given byte cap and compare against the
// reference splitter
void check(std::string const& content, size_t const max_bytes, Communicator const& comm) {
    auto const path = write_shared_file(content, comm);
    auto const lines = read_all_lines(path, max_bytes, comm);
    EXPECT_EQ(lines, expected_lines(content, max_bytes)) << "max_bytes=" << max_bytes;
    remove_shared_file(path, comm);
}

constexpr size_t line_width = 3;                  // lines "000".."239"
constexpr size_t line_count = 240;                // record = line_width + 1 newline = 4 bytes
constexpr size_t record_size = line_width + 1;

TEST(FileGenerator, ReadsEveryLineInOrder) {
    Communicator comm;
    check(make_content(line_count, line_width), /*max_bytes=*/0, comm);
}

TEST(FileGenerator, MaxNumBytesZeroReadsWholeFile) {
    Communicator comm;
    auto const content = make_content(line_count, line_width);
    check(content, /*max_bytes=*/0, comm);
}

TEST(FileGenerator, MaxNumBytesLargerThanFileReadsWholeFile) {
    Communicator comm;
    auto const content = make_content(line_count, line_width);
    check(content, /*max_bytes=*/10 * content.size(), comm);
}

TEST(FileGenerator, MaxNumBytesTruncatesAtLineBoundary) {
    Communicator comm;
    // a cut right after the 100th newline: the first 100 lines, nothing partial
    check(make_content(line_count, line_width), /*max_bytes=*/100 * record_size, comm);
}

TEST(FileGenerator, MaxNumBytesTruncatesMidLine) {
    Communicator comm;
    // two bytes into line 100 ("100"): the first 100 lines plus the truncated fragment "10"
    check(make_content(line_count, line_width), /*max_bytes=*/100 * record_size + 2, comm);
}

TEST(FileGenerator, MaxNumBytesSmallerThanOneLine) {
    Communicator comm;
    // the cut lands inside the very first line: a single truncated string "00"
    check(make_content(line_count, line_width), /*max_bytes=*/2, comm);
}

} // namespace
