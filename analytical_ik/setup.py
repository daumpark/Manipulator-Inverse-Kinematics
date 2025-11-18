from setuptools import setup
package_name = 'analytical_ik'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/planar2d.urdf']),
        ('share/' + package_name + '/launch', [
            'launch/analytical_ik_test.launch.py',
        ]),
        ('share/' + package_name + '/rviz', [
            'rviz/planar2d.rviz',
        ]),
    ],
    install_requires=['setuptools','numpy'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Analytical IK for 2D planar and PiPER (Python)',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'ik_node = analytical_ik.nodes.ik_node:main',
        ],
    },
)
