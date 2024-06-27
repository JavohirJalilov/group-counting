fps = 30

data = {1: 
        {
            "start": 245, 
            "end": 350
        }
    }

time_spent = (data[1]["end"] - data[1]["start"]) / fps

print(f"time spent: {time_spent}s")

print(1 in data)