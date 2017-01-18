#pragma once

#include <cmath>

// activation class
class ActivationAbst
{
public:
	virtual float activation(float in)  const = 0;
	virtual float calc_deriv(float in) const = 0;
};
// sigmoid
class SigmoidActivation : public ActivationAbst
{
public:
	virtual float activation(float in) const override
	{
		return 1 / (1 + std::exp(-in));
	}
	virtual float calc_deriv(float in) const override
	{
		auto act = activation(in);
		return act * (1 - act);
	}
};
// ReLU
class ReLUActivation : public ActivationAbst
{
public:
	virtual float activation(float in) const override
	{
		return in >= 0 ? in : 0;
	}
	virtual float calc_deriv(float in) const override
	{
		return in > 0 ? 1 : 0;
	}
};
// P“™Ê‘œ
class IdentityActivation : public ActivationAbst
{
public:
	virtual float activation(float in) const override
	{
		return in;
	}
	virtual float calc_deriv(float in) const override
	{
		return 1;
	}
};
