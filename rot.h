__device__ static inline uint16_t rol16(uint16_t value, size_t rot)
{
    return (value << rot) | (value >> (16 - rot));
}

__device__ static inline uint16_t ror16(uint16_t value, size_t rot)
{
    return (value >> rot) | (value << (16 - rot));
}

__device__ static inline uint32_t rol32(uint32_t value, size_t rot)
{
    return (value << rot) | (value >> (32 - rot));
}

__device__ static inline uint32_t ror32(uint32_t value, size_t rot)
{
    return (value >> rot) | (value << (32 - rot));
}
