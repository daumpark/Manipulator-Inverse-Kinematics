from setuptools import setup
package_name = 'heuristic_ik'
setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/heuristic_ik_test.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/piper.rviz']),
    ],
    install_requires=['setuptools','pinocchio','scipy'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Heuristic IK (FABRIK3D, FABRIK2D, CCD2D)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'ik_node = heuristic_ik.nodes.ik_node:main',
        ],
    },
)
