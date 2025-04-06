echo "source /opt/ros/jazzy/setup.bash" >> ~root/.bashrc

# Temporary install host vpi
sudo apt install gnupg
sudo apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
sudo apt install -y software-properties-common
sudo add-apt-repository -y 'deb https://repo.download.nvidia.com/jetson/x86_64/jammy r36.4 main'
sudo apt update
sudo apt install libnvvpi3 vpi3-dev vpi3-samples

# Testing COLMAP
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

# Fixing bug?
# sudo apt-get install gcc-10 g++-10
# export CC=/usr/bin/gcc-10
# export CXX=/usr/bin/g++-10
# export CUDAHOSTCXX=/usr/bin/g++-10

git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j5