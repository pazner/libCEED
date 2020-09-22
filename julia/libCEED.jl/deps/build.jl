const pathkey = "JULIA_LIBCEED_LIB"

if isfile("config.txt")
   rm("config.txt")
end

if haskey(ENV, pathkey)
   ceedpath = ENV[pathkey]
   open("config.txt", write=true) do f
      println(f, ceedpath)
   end
else
   error("""
   JULIA_LIBCEED_LIB environment variable not set.

   To build libCEED.jl, set the JULIA_LIBCEED_LIB environment variable
   to the absolute path of the libCEED dynamic library.
   """)
end
