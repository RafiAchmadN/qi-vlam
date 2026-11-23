from glob import glob

from setuptools import setup

package_name = "qi_vla_ros"

setup(
    name=package_name,
    version="1.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="QI-VLA contributors",
    description="QI-VLA arm: URDF, simulation node and policy node for ROS 2",
    license="MIT",
    entry_points={
        "console_scripts": [
            "qi_vla_sim = qi_vla_ros.sim_node:main",
            "qi_vla_policy = qi_vla_ros.policy_node:main",
            "qi_vla_rover_sim = qi_vla_ros.rover_node:main",
        ],
    },
)
