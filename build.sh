build_path="./build"
if [ -d "$build_path"]; then 
    rm -rf $build_path
fi 

mkdir $build_path
cd $build_path
cmake .. -DEIGEN3_INCLUDE_DIR=./third_party/include -DENABLE_CPP_EXAMPLES=ON

make -j 14

# Test with an example
./examples/xor