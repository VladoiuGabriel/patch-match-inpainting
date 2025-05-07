# Image Inpainting – Patch-Based Method (C++ with OpenCV)

This project implements a traditional image inpainting algorithm in C++ using OpenCV, without relying on object-oriented programming. It reconstructs missing or damaged image regions by copying patches from known areas based on similarity and structural coherence.

## Features

- Fully procedural implementation (no C++ classes used)
- Patch selection based on color similarity and edge continuity
- Manual mask input via mouse interaction
- Compatible with OpenCV 4.x

## Project Structure

- `main.cpp` – handles user interaction and runs the inpainting process
- `inpaint.cpp` – core algorithm implementation
- `inpaint.h` – structure definitions and function declarations
- `Lincoln.jpg` – sample image for testing

## Requirements

- CMake ≥ 3.10
- OpenCV ≥ 4.x
- C++17-compatible compiler



## Usage

1. Run the program.
2. Mark the region to be inpainted using the mouse.Pressing "+" or "-" changes the cursor's size.
3. Press `i` or space to start the inpainting process.
4. Press `e` to exit or `o` to reset the mask.


