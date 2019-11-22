module Metrics

using ..Photon: MetricStore, stop, Workout, Tensor, getmetricvalue, saveWorkout, getContext
using Printf
using Statistics
import Knet

include("core.jl")
export getmetricname, SmartReducer, update!, history, BinaryAccuracy, OneHotBinaryAccuracy

include("meters.jl")
export Meter, ConsoleMeter, SilentMeter, TensorBoardMeter, FileMeter, PlotMeter

include("callbacks.jl")
export AutoSave, EpochSave, EarlyStop

end
