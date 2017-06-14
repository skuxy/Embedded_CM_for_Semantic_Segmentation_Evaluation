sudo systemctl stop network-manager
sudo ip link set eth0 down
sudo ip addr add 192.168.1.2/24 broadcast 192.168.1.255 dev eth0
sudo ip link set eth0 up
sudo ip route add 191.168.1.1 dev eth0
sudo ip route add default via 191.168.1.1 dev eth0
sudo systemctl restart isc-dhcp-server.service
