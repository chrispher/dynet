prefix=`dirname "$0"`
prefix=`cd "$prefix"; pwd`

build_path="packages"
if [ ! -d "$build_path" ]; then 
	mkdir "$build_path" 
fi 

cd $build_path

filename="eigen-eigen-5a0156e40feb"
eigen_name="3.3.4.tar.bz2"
if [ ! -f "$eigen_name" ]; then 
	wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
fi 

tar -jxf $eigen_name

# build Eigen
mkdir -p $prefix/lib
mkdir -p $prefix/include
rm -rf $prefix/include/Eigen
rm -rf $prefix/include/unsupported
cd $filename && mv Eigen ../../include && mv unsupported ../../include && cd ..

echo "Done!"

