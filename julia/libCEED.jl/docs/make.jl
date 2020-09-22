using Documenter, libCEED, LinearAlgebra

makedocs(sitename="libCEED.jl Docs",
         format=Documenter.HTML(prettyurls=false),
         pages=[
             "Home" => "index.md",
             "Ceed Objects" => [
                "Ceed.md",
                "CeedVector.md",
                "ElemRestriction.md",
                "Basis.md",
                "QFunction.md",
                "Operator.md",
             ],
             "Utilities" => [
                "Misc.md",
                "Globals.md",
                "Quadrature.md",
             ],
             "C.md",
             "UserQFunctions.md",
             "Examples.md",
         ])
