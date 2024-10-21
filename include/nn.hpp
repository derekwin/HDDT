#ifndef NN_H
#define NN_H

namespace nn
{
    void InitDriver(int index);
    void InitArray(int, float*, int, float);
    void AddArray(int, float*, float*, int);
    void MemcpyDeviceToHost(int, float*, float*);
}

#endif