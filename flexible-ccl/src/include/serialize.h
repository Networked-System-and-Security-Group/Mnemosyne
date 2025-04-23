#ifndef NCCL_SERIALIZE_H_
#define NCCL_SERIALIZE_H_

#include "bootstrap.h"
#include "scale.h"
#include "transport.h"
#include "graph/topo.h"

#define SERIAL_CONVERT(DST, SRC) reinterpret_cast<decltype(DST)>(SRC)
#define SERIAL_ASSIGN(DST, SRC) ((DST) = SERIAL_CONVERT(DST, SRC))

template <typename info_t>
inline size_t ncclInfoSerializeBase(char *buffer, const info_t *info) {
  memcpy(buffer, info, sizeof(info_t));
  return sizeof(info_t);
}

template <typename info_t>
inline size_t ncclInfoSerialize(char *buffer, const info_t *info) {
  return ncclInfoSerializeBase(buffer, info);
}

template <typename T>
inline size_t ncclArraySerialize(char *buffer, const T *arr, size_t n) {
  size_t offset = 0;
  for (size_t i = 0; i < n; i++) {
    offset += ncclInfoSerialize(buffer + offset, arr + i);
  }
  return offset;
}

template <typename info_t>
inline size_t ncclInfoDeserializeBase(info_t *info) {
  return sizeof(info_t);
}

template <typename info_t>
inline size_t ncclInfoDeserialize(info_t *info) {
  return ncclInfoDeserializeBase(info);
}

template <typename T>
inline size_t ncclArrayDeserialize(T *arr, size_t n) {
  size_t offset = 0;
  char *buffer = (char *)arr;
  for (size_t i = 0; i < n; i++) {
    offset += ncclInfoDeserialize((T *)(buffer + offset));
  }
  return offset;
}

template <>
inline size_t ncclInfoSerialize(char *buffer, const struct bootstrapState *info) {
  size_t offset = 0;
  offset += ncclInfoSerializeBase(buffer + offset, info);
  offset += ncclArraySerialize(buffer + offset, info->peerProxyAddressesUDS, info->nranks);
  offset += ncclArraySerialize(buffer + offset, info->peerProxyAddresses, info->nranks);
  offset += ncclArraySerialize(buffer + offset, info->peerP2pAddresses, info->nranks);
  return offset;
}

template <>
inline size_t ncclInfoDeserialize(struct bootstrapState *info) {
  size_t offset = 0;
  char *buffer = (char *)info;
  offset += ncclInfoDeserializeBase(info);
  offset += ncclArrayDeserialize(SERIAL_ASSIGN(info->peerProxyAddressesUDS, buffer + offset), info->nranks);
  offset += ncclArrayDeserialize(SERIAL_ASSIGN(info->peerProxyAddresses, buffer + offset), info->nranks);
  offset += ncclArrayDeserialize(SERIAL_ASSIGN(info->peerP2pAddresses, buffer + offset), info->nranks);
  return offset;
}

template <>
inline size_t ncclInfoSerialize(char *buffer, const ncclComm *info) {
  size_t offset = 0;
  offset += ncclInfoSerializeBase(buffer + offset, info);
  offset += ncclInfoSerialize(buffer + offset, static_cast<bootstrapState *>(info->bootstrap));
  if (info->peerInfo)
    offset += ncclArraySerialize(buffer + offset, info->peerInfo, info->nRanks + 1);
  // if (info->topo)
  //   offset += ncclInfoSerialize(buffer + offset, info->topo);
  if (info->rankToNode)
    offset += ncclArraySerialize(buffer + offset, info->rankToNode, info->nRanks);
  return offset;
}

template <>
inline size_t ncclInfoDeserialize(struct ncclComm *info) {
  size_t offset = 0;
  char *buffer = (char *)info;
  offset += ncclInfoDeserializeBase(info);
  offset += ncclInfoDeserialize((bootstrapState *)SERIAL_ASSIGN(info->bootstrap, buffer + offset));
  if (info->peerInfo)
    offset += ncclArrayDeserialize(SERIAL_ASSIGN(info->peerInfo, buffer + offset), info->nRanks + 1);
  // if (info->topo)
  //   offset += ncclInfoDeserialize(SERIAL_ASSIGN(info->topo, buffer + offset));
  if (info->rankToNode)
    offset += ncclArrayDeserialize(SERIAL_ASSIGN(info->rankToNode, buffer + offset), info->nRanks);
  return offset;
}

template <>
inline size_t ncclInfoSerialize(char *buffer, const ncclCommInfoInternal *info) {
  size_t offset = 0;
  offset += ncclInfoSerializeBase(buffer + offset, info);
  offset += ncclInfoSerialize(buffer + offset, info->comm);
  offset += ncclInfoSerialize(buffer + offset, info->uniqueId);
  return offset;
}

template <>
inline size_t ncclInfoDeserialize(struct ncclCommInfoInternal *info) {
  size_t offset = 0;
  char *buffer = (char *)info;
  offset += ncclInfoDeserializeBase(info);
  offset += ncclInfoDeserialize(SERIAL_ASSIGN(info->comm, buffer + offset));
  offset += ncclInfoDeserialize(SERIAL_ASSIGN(info->uniqueId, buffer + offset));
  return offset;
}

template <>
inline size_t ncclInfoSerialize(char *buffer, const ncclNewRankInfoInternal *info) {
  size_t offset = 0;
  offset += ncclInfoSerializeBase(buffer + offset, info);
  offset += ncclInfoSerialize(buffer + offset, info->comm);
  return offset;
}

template <>
inline size_t ncclInfoDeserialize(struct ncclNewRankInfoInternal *info) {
  size_t offset = 0;
  char *buffer = (char *)info;
  offset += ncclInfoDeserializeBase(info);
  offset += ncclInfoDeserialize(SERIAL_ASSIGN(info->comm, buffer + offset));
  return offset;
}

#endif // NCCL_SERIALIZE_H_
