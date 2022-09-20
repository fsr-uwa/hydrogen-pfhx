# simple test for verifying model
import matplotlib.pyplot as plt    
from hydrogen_pfhx import model, outputs

# run the model with alt config
results = model.model('src/configs/alternate_configuration.yaml')

# print results
print(results)

# plot & display results!
outputs.plot_results(results)
plt.show()