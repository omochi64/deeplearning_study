#pragma once

#include <cmath>

// activation class
class ActivationAbst
{
public:
	virtual double activation(double in)  const = 0;
	virtual double calc_deriv(double in) const = 0;
};
// sigmoid
class SigmoidActivation : public ActivationAbst
{
public:
	virtual double activation(double in) const override
	{
		return 1 / (1 + std::exp(-in));
	}
	virtual double calc_deriv(double in) const override
	{
		auto act = activation(in);
		return act * (1 - act);
	}
};
// ReLU
class ReLUActivation : public ActivationAbst
{
public:
	virtual double activation(double in) const override
	{
		return in >= 0 ? in : 0;
	}
	virtual double calc_deriv(double in) const override
	{
		return in > 0 ? 1 : 0;
	}
};
// P“™Ê‘œ
class IdentityActivation : public ActivationAbst
{
public:
	virtual double activation(double in) const override
	{
		return in;
	}
	virtual double calc_deriv(double in) const override
	{
		return 1;
	}
};
