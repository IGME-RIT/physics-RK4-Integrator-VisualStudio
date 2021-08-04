/*
Title: Runge-Kutta 4 integration method
File Name: integrator.cpp
Copyright © 2015
Original authors: Srinivasan Thiagarajan
Written under the supervision of David I. Schwartz, Ph.D., and
supported by a professional development seed grant from the B. Thomas
Golisano College of Computing & Information Sciences
(https://www.rit.edu/gccis) at the Rochester Institute of Technology.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Description:
This is a program that shows the difference in integration mechanisms Runge kutta and
Euler integration method. The runge kutta method is of fourth order. Euler integrator,
takes the value of the function F(x,y(x)) and integrates it over the time step T.
Unlike Euler integrator, RK integrator, divides the entire time step into 3 parts and 
averages the slope of the velocity It uses that value of the y(x) to integrate over
the entire time step T. This results in having a reduced margin of error.

You can see the error margin between the two techniques. The red line uses the RK method,
while the blue line uses euler integrator. The red line is closer to its precise implementation.

use "space" to move one time Step.

References:
Nicholas Gallagher
Book : physics based animation by Kenny Erleben,Jon Sporring, Knud Henriksen and Henrik ;
Wikipedia : https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods ;
https://en.wikipedia.org/wiki/Midpoint_method

*/

#pragma once
#include "GLIncludes.h"

typedef float(*TwoVarFunc)(float, float);



// Eulers Method is a first order numerical integration method that integrates an IVP ODE in the form dx/dt = f(x, t), x(a) = b.
// dxdt is the function that we are taking the numerical integral of, so f(x, t).
// initialValue is the x and t values of the initial value of the IVP, so <a, b>.
// tValue is the value that we are trying to approximate. The output of this function will be an approximation of <tValue, x(tValue)>.
// steps is the number of steps to iterate over. The higher the number, the more accurate, but slower.
glm::vec2 EulersMethod(TwoVarFunc dxdt, glm::vec2 initialValue, float tValue, int steps) {
	
	// Setup the array to store the data.
	glm::vec2* computedValues = (glm::vec2*) malloc(sizeof(glm::vec2) * (steps + 1));
	computedValues[0] = initialValue;

	// Calculate the distance of each step.
	float stepDistance = (tValue - initialValue[1]) / steps;

	// Loop through until we get to the wanted tValue.
	for (int i = 1; i <= steps; i++) {
		// Get the values to work with.
		const glm::vec2& previous = computedValues[i - 1];
		float h = previous[0] + stepDistance;

		// Apply the Euler formula and store.
		computedValues[i] = glm::vec2(h, previous[0] + h * (dxdt(previous[0], previous[1])));
	}

	// Return the final value.
	glm::vec2 returnedValue = computedValues[steps];
	free(computedValues);
	return returnedValue;
}



// Runge-Kutta 4 is a fourth order numerical integration method that integrates an IVP ODE in the form dx/dt = f(x, t), x(a) = b.
// dxdt is the function that we are taking the numerical integral of, so f(x, t).
// initialValue is the x and t values of the initial value of the IVP, so <a, b>.
// tValue is the value that we are trying to approximate. The output of this function will be an approximation of <tValue, x(tValue)>.
// steps is the number of steps to iterate over. The higher the number, the more accurate, but slower.
glm::vec2 RK4(TwoVarFunc dxdt, glm::vec2 initialValue, float tValue, int steps) {
	
	// Setup the array to store the data.
	glm::vec2* computedValues = (glm::vec2*) malloc(sizeof(glm::vec2) * (steps + 1));
	computedValues[0] = initialValue;

	// Calculate the distance of each step.
	float stepDistance = (tValue - initialValue[1]) / steps;

	// Loop through until we get to the wanted tValue.
	for (int i = 1; i <= steps; i++) {
		// Get the values to work with.
		const glm::vec2& previous = computedValues[i-1];
		float h = previous[0] + stepDistance;

		// Apply the Runge-Kutta formula.
		float k1 = h * dxdt(previous[0], previous[1]);
		float k2 = h * dxdt(previous[0] + (k1 / 2.0f), previous[1] + (h / 2.0f));
		float k3 = h * dxdt(previous[0] + (k2 / 2.0f), previous[1] + (h / 2.0f));
		float k4 = h * dxdt(previous[0] + k3, previous[1] + h);

		computedValues[i] = glm::vec2(h, previous[0] + (1.0f / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4));
	}

	// Return the final value, clean up memory.
	glm::vec2 returnedValue = computedValues[steps];
	free(computedValues);
	return returnedValue;
}


//
// If you would like to just do one step of integration, you can use the following for less overhead.
//

// Actual eulers: x(n*h) = x([n-1]*h) + h * dydx([n-1]*h, x([n-1]*h))
// Function pointer dxdt: function f(x,t)
// vec2 previous: most recent value of function for x and t (e.g. x([n-1]*h), [n-1]*h)
// float step: the amount to step by (e.g. h)
float EulersMethodIteration(TwoVarFunc dxdt, glm::vec2 previous, float step) {
	return previous[0] + step * (dxdt(previous[0], previous[1]));
}

// Function pointer dxdt: function f(x,t)
// vec2 previous: most recent value of function for x and t (e.g. x([n-1]*h), [n-1]*h)
// float step: the amount to step by (e.g. h)
float RungaKutta4Iteration(TwoVarFunc dxdt, glm::vec2 previous, float step) {
	
	// Find the next value of x (e.g. x(n*h))
	float k1 = step * dxdt(previous[0], previous[1]);
	float k2 = step * dxdt(previous[0] + (k1 / 2.0f), previous[1] + (step / 2.0f));
	float k3 = step * dxdt(previous[0] + (k2 / 2.0f), previous[1] + (step / 2.0f));
	float k4 = step * dxdt(previous[0] + k3, previous[1] + step);

 	return previous[0] + (1.0f / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
}