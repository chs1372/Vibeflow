#include "io/VtuWriter.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vibeflow {
namespace {

const char* B64 =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64(const unsigned char* p, std::size_t n) {
  std::string out;
  out.reserve(((n + 2) / 3) * 4);
  std::size_t i = 0;
  for (; i + 2 < n; i += 3) {
    const unsigned v = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
    out += B64[(v >> 18) & 63]; out += B64[(v >> 12) & 63];
    out += B64[(v >> 6) & 63];  out += B64[v & 63];
  }
  if (i < n) {
    unsigned v = p[i] << 16;
    const bool two = (i + 1 < n);
    if (two) v |= p[i + 1] << 8;
    out += B64[(v >> 18) & 63];
    out += B64[(v >> 12) & 63];
    out += two ? B64[(v >> 6) & 63] : '=';
    out += '=';
  }
  return out;
}

// VTK's base64 DataArray payload is: base64(header) followed by base64(data),
// where header is the byte count as a UInt32. The two are encoded separately.
template <class T>
std::string encodeArray(const std::vector<T>& v) {
  const std::uint32_t nbytes = static_cast<std::uint32_t>(v.size() * sizeof(T));
  std::string out = base64(reinterpret_cast<const unsigned char*>(&nbytes), 4);
  out += base64(reinterpret_cast<const unsigned char*>(v.data()), nbytes);
  return out;
}

template <class T>
void dataArray(std::ostream& os, const char* type, const std::string& name,
               int ncomp, const std::vector<T>& v) {
  os << "        <DataArray type=\"" << type << "\" Name=\"" << name << "\"";
  if (ncomp > 1) os << " NumberOfComponents=\"" << ncomp << "\"";
  os << " format=\"binary\">\n          " << encodeArray(v) << "\n        </DataArray>\n";
}

}  // namespace

VtuWriter::VtuWriter(const std::vector<Vec3>& points,
                     const std::vector<std::array<Index, 8>>& hexes)
    : points_(points), hexes_(hexes) {}

void VtuWriter::addCellField(const std::string& name, const ScalarField& f) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), f);
  Field fl{name, 1, {}};
  fl.data.resize(h.extent(0));
  for (std::size_t i = 0; i < h.extent(0); ++i) fl.data[i] = h(i);
  fields_.push_back(std::move(fl));
}

void VtuWriter::addCellField(const std::string& name, const VectorField& f) {
  auto h = Kokkos::create_mirror_view_and_copy(HostSpace::memory_space(), f);
  Field fl{name, 3, {}};
  fl.data.resize(h.extent(0) * 3);
  for (std::size_t i = 0; i < h.extent(0); ++i)
    for (int k = 0; k < 3; ++k) fl.data[i * 3 + k] = h(i, k);
  fields_.push_back(std::move(fl));
}

std::string VtuWriter::writePiece(const std::string& path) const {
  std::ofstream os(path);
  if (!os) throw std::runtime_error("cannot write " + path);

  std::vector<Real> xyz(points_.size() * 3);
  for (std::size_t i = 0; i < points_.size(); ++i) {
    xyz[i * 3 + 0] = points_[i].x; xyz[i * 3 + 1] = points_[i].y; xyz[i * 3 + 2] = points_[i].z;
  }
  std::vector<std::int64_t> conn(hexes_.size() * 8), offs(hexes_.size());
  std::vector<std::uint8_t> types(hexes_.size(), 12);   // 12 = VTK_HEXAHEDRON
  for (std::size_t c = 0; c < hexes_.size(); ++c) {
    for (int t = 0; t < 8; ++t) conn[c * 8 + t] = hexes_[c][t];
    offs[c] = static_cast<std::int64_t>((c + 1) * 8);
  }

  os << "<?xml version=\"1.0\"?>\n"
     << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\""
     << " header_type=\"UInt32\">\n"
     << "  <UnstructuredGrid>\n"
     << "    <Piece NumberOfPoints=\"" << points_.size()
     << "\" NumberOfCells=\"" << hexes_.size() << "\">\n"
     << "      <Points>\n";
  dataArray(os, "Float64", "Points", 3, xyz);
  os << "      </Points>\n      <Cells>\n";
  dataArray(os, "Int64", "connectivity", 1, conn);
  dataArray(os, "Int64", "offsets", 1, offs);
  dataArray(os, "UInt8", "types", 1, types);
  os << "      </Cells>\n      <CellData>\n";
  for (const auto& f : fields_) dataArray(os, "Float64", f.name, f.ncomp, f.data);
  os << "      </CellData>\n    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n";
  return path;
}

std::string VtuWriter::write(const std::string& basename) const {
  return writePiece(basename + ".vtu");
}

std::string VtuWriter::pvtuContents(const std::string& basename, int nRanks) const {
  std::ostringstream os;
  const auto slash = basename.find_last_of('/');
  const std::string stem = slash == std::string::npos ? basename : basename.substr(slash + 1);
  os << "<?xml version=\"1.0\"?>\n"
     << "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
     << "  <PUnstructuredGrid GhostLevel=\"0\">\n"
     << "    <PPoints>\n"
     << "      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\"/>\n"
     << "    </PPoints>\n    <PCellData>\n";
  for (const auto& f : fields_) {
    os << "      <PDataArray type=\"Float64\" Name=\"" << f.name << "\"";
    if (f.ncomp > 1) os << " NumberOfComponents=\"" << f.ncomp << "\"";
    os << "/>\n";
  }
  os << "    </PCellData>\n";
  for (int r = 0; r < nRanks; ++r)
    os << "    <Piece Source=\"" << stem << "_" << r << ".vtu\"/>\n";
  os << "  </PUnstructuredGrid>\n</VTKFile>\n";
  return os.str();
}

std::string VtuWriter::writeParallel(const std::string& basename,
                                     int rank, int nRanks) const {
  const std::string piece = basename + "_" + std::to_string(rank) + ".vtu";
  writePiece(piece);
  if (rank == 0) {
    std::ofstream os(basename + ".pvtu");
    os << pvtuContents(basename, nRanks);
    return basename + ".pvtu";
  }
  return piece;
}

}  // namespace vibeflow
