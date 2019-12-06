module Metrics

using ..Photon: MetricStore, Workout, Tensor, getmetricvalue, getContext
using Printf
using Statistics
import Knet

include("core.jl")
export getmetricname, SmartReducer, update!, history, BinaryAccuracy, OneHotBinaryAccuracy


end
