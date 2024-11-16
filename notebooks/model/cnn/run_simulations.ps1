# PowerShell script to run each simulation encoding in parallel
Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bad-player-3-bootstrap-simulation"
Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bootstrap-p3-010-simulation"
Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bootstrap-simulation-p0d05-p0d25"
Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bootstrap-simulation-p0d15"
Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bootstrap-simulation"
Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bootstrap-simulation-short"
Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bootstrap-simulation-random-noise"

# Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "dql-cnn-v002-vs-bootstrap-p0-simulation"
# Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "dql-cnn-v002-vs-bootstrap-p0-simulation"
# Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "dql-cnn-v002-vs-bootstrap-p0-simulation"
# Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "dql-cnn-v003-vs-bootstrap-p0-simulation"
# Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bootstrap-0d01-vs-dql-v004"
# Start-Process -FilePath "python" -ArgumentList "encode_simulations.py", "bootstrap-vs-dql-v003"

Write-Output "All simulations started in parallel."
