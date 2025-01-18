import React, { useState } from "react";

const ConnectWallet = () => {
  const [walletAddress, setWalletAddress] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [backendResponse, setBackendResponse] = useState(null);

  const connectWallet = async () => {
    if (typeof window.ethereum !== "undefined") {
      try {
        const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
        const address = accounts[0];
        setWalletAddress(address);
        setErrorMessage(null);

        // Send the wallet address to the backend
        const response = await fetch('${process.env.REACT_APP_API_URL}/connectwallet', {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ walletAddress: address }),
        });

        if (response.ok) {
          const data = await response.json();
          setBackendResponse(data.message);
        } else {
          const error = await response.json();
          setBackendResponse(error.error);
        }
      } catch (error) {
        setErrorMessage("Error connecting to the wallet. Please try again.");
        console.error(error);
      }
    } else {
      setErrorMessage("No crypto wallet found. Please install MetaMask!");
    }
  };

  return (
    <div>
     
      {walletAddress ? (
        <p>
          <strong>Connected Wallet Address:</strong> {walletAddress}
        </p>
      ) : (
        <button onClick={connectWallet} className="connect-wallet-button">
          Connect Wallet
        </button>
      )}

      {/* {errorMessage && <p style={{ color: "red" }}>{errorMessage}</p>} */}
      {backendResponse && <p>{backendResponse}</p>}
    </div>
  );
};

export default ConnectWallet;
