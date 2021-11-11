function velocity(t::Array{Float64,1}; A::Number = 1, f::Number = 0.5)
	@. A * 1/(2 * π * f) * (1 - cos(2 * π * f * t))
end
    
function acceleration(t::Array{Float64,1}; A::Number = 1, f::Number = 0.5)
	@. A * sin(2 * π * f * t)
end
    
function displacement(A::Number; T::Number = 1)	
	A * T^2 /(2π)
end 
    
function amplitude(θ::Number; T::Number = 1)	
	(2π/T^2) * θ
end
    
function peak_velocity(A::Number; T::Number = 1)	
	(T/π) * A
end