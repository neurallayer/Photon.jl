module Metrics

using ..Photon: MetricStore, stop, Workout, Tensor, Meter, getmetricvalue, saveWorkout, getContext
using Printf
using Statistics
import Knet

include("core.jl")
export SmartReducer, update!, history, BinaryAccuracy, OneHotBinaryAccuracy

include("meters.jl")
export ConsoleMeter, TensorBoardMeter, FileMeter, PlotMeter

include("callbacks.jl")
export AutoSave, EarlyStop

end
