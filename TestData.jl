module BDTTestData

using DataFrames

export test_df

test_df = DataFrame(F1=[0,1,1,1,1,1,1], F2=[1,0,1,0,1,0,1], F3=[1,1,0,0,0,1,1], OUT=[0,1,1,0,0,1,1])

end