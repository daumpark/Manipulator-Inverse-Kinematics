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
            'launch/analytical_2d_test.launch.py',
            'launch/analytical_piper_test.launch.py'
        ]),
        ('share/' + package_name + '/rviz', [
            'rviz/planar2d.rviz',
            'rviz/piper.rviz'
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
            'ik_node_2d = analytical_ik.nodes.ik_node_2d:main',
            'ik_node_piper = analytical_ik.nodes.ik_node_piper:main',
        ],
    },
)
