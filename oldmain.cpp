#include <vector>

#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/container/vector.hpp>

namespace compute = boost::compute;

int main()
{
    // get default device and setup context
    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device);

    // create data array on host
    int host_data[] = { 1, 3, 5, 7, 9 };

    // create vector on device
    compute::vector<int> device_vector(5, context);

    // copy from host to device
    compute::copy(
            host_data, host_data + 5, device_vector.begin(), queue
    );

    // create vector on host
    std::vector<int> host_vector(5);

    // copy data back to host
    compute::copy(
            device_vector.begin(), device_vector.end(), host_vector.begin(), queue
    );

    return 0;
}

