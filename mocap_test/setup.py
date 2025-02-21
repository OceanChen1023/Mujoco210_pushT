from setuptools import setup

package_name = 'mocap_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ocean',
    maintainer_email='pig841023@gmail.com',
    description='Examples of minimal publisher/subscriber using rclpy',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mocap_control_pole = mocap_test.mocap_control_pole_ros2:main',  #/python_file_Path/py_file/main_function
            'listener = mocap_test.subscriber_member_function:main',
            'talker = mocap_test.publisher_member_function:main',
        ],
    },
)
