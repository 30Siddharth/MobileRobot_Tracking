# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/team4/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/team4/catkin_ws/build

# Utility rule file for octomap_server_gencfg.

# Include the progress variables for this target.
include experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/progress.make

experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg: /home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h
experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg: /home/team4/catkin_ws/devel/lib/python2.7/dist-packages/octomap_server/cfg/OctomapServerConfig.py


/home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h: /home/team4/catkin_ws/src/experimental_robotics_course/src/octomap_mapping/octomap_server/cfg/OctomapServer.cfg
/home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h: /opt/ros/melodic/share/dynamic_reconfigure/templates/ConfigType.py.template
/home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h: /opt/ros/melodic/share/dynamic_reconfigure/templates/ConfigType.h.template
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/team4/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating dynamic reconfigure files from cfg/OctomapServer.cfg: /home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h /home/team4/catkin_ws/devel/lib/python2.7/dist-packages/octomap_server/cfg/OctomapServerConfig.py"
	cd /home/team4/catkin_ws/build/experimental_robotics_course/src/octomap_mapping/octomap_server && ../../../../catkin_generated/env_cached.sh /home/team4/catkin_ws/build/experimental_robotics_course/src/octomap_mapping/octomap_server/setup_custom_pythonpath.sh /home/team4/catkin_ws/src/experimental_robotics_course/src/octomap_mapping/octomap_server/cfg/OctomapServer.cfg /opt/ros/melodic/share/dynamic_reconfigure/cmake/.. /home/team4/catkin_ws/devel/share/octomap_server /home/team4/catkin_ws/devel/include/octomap_server /home/team4/catkin_ws/devel/lib/python2.7/dist-packages/octomap_server

/home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig.dox: /home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h
	@$(CMAKE_COMMAND) -E touch_nocreate /home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig.dox

/home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig-usage.dox: /home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h
	@$(CMAKE_COMMAND) -E touch_nocreate /home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig-usage.dox

/home/team4/catkin_ws/devel/lib/python2.7/dist-packages/octomap_server/cfg/OctomapServerConfig.py: /home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h
	@$(CMAKE_COMMAND) -E touch_nocreate /home/team4/catkin_ws/devel/lib/python2.7/dist-packages/octomap_server/cfg/OctomapServerConfig.py

/home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig.wikidoc: /home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h
	@$(CMAKE_COMMAND) -E touch_nocreate /home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig.wikidoc

octomap_server_gencfg: experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg
octomap_server_gencfg: /home/team4/catkin_ws/devel/include/octomap_server/OctomapServerConfig.h
octomap_server_gencfg: /home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig.dox
octomap_server_gencfg: /home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig-usage.dox
octomap_server_gencfg: /home/team4/catkin_ws/devel/lib/python2.7/dist-packages/octomap_server/cfg/OctomapServerConfig.py
octomap_server_gencfg: /home/team4/catkin_ws/devel/share/octomap_server/docs/OctomapServerConfig.wikidoc
octomap_server_gencfg: experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/build.make

.PHONY : octomap_server_gencfg

# Rule to build all files generated by this target.
experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/build: octomap_server_gencfg

.PHONY : experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/build

experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/clean:
	cd /home/team4/catkin_ws/build/experimental_robotics_course/src/octomap_mapping/octomap_server && $(CMAKE_COMMAND) -P CMakeFiles/octomap_server_gencfg.dir/cmake_clean.cmake
.PHONY : experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/clean

experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/depend:
	cd /home/team4/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/team4/catkin_ws/src /home/team4/catkin_ws/src/experimental_robotics_course/src/octomap_mapping/octomap_server /home/team4/catkin_ws/build /home/team4/catkin_ws/build/experimental_robotics_course/src/octomap_mapping/octomap_server /home/team4/catkin_ws/build/experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : experimental_robotics_course/src/octomap_mapping/octomap_server/CMakeFiles/octomap_server_gencfg.dir/depend
