from dspy import Example 

class ClaimDemo:
    def createExample(self):
        f=open("C:/Users/user/Python/dspy/src/multi-hop/data/claimexample.txt")
        ex=[]
        for l in f.readlines():
            claimid=l.split(" ",1)
            ex.append(Example(question="What happened to my claim "+claimid[0], answer=claimid[1]).with_inputs("question"))
        
        #print(ex)
        return ex
    def getData(self):
        f=open("C:/Users/user/Python/dspy/src/multi-hop/data/claimsource_react.txt")
        data={}
        ldata=[]
        lids=[]
        for i,l in enumerate(f.readlines()):
            claimid=l.split(" ",1)
            ldata.append(l)
            lids.append(f"ids{i}")
            data[claimid[0]]=l
            
        #print(lids)
        #print(ldata)
        return ldata, lids
if __name__=="__main__":

    cr=ClaimDemo()
    cr.getData()

