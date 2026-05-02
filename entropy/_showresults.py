import dnest4
f = open("logzs.txt", "a")
result = dnest4.postprocess(plot=False)[0]
f.write(str(result) + "\n")
f.close()
