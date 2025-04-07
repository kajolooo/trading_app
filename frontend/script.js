// const SYMBOL = 'AAPL'; // Removed hardcoded symbol
const DEFAULT_QTY = 10; // Default quantity for paper trades (kept for potential future use)

// Get references to DOM elements
const symbolsTextarea = document.getElementById('symbols-textarea'); // Changed from symbolInput
const fetchBtn = document.getElementById('fetch-btn');
const buyingPowerSpan = document.getElementById('buying-power');
const resultsTbody = document.getElementById('results-tbody'); // Added table body reference
const errorMessageDiv = document.getElementById('error-message');

// --- Helper Functions ---

// Function to display general error messages
function displayError(message) {
    if (errorMessageDiv) {
        errorMessageDiv.textContent = message;
    } else {
        console.error("Error display element #error-message not found!");
        if (message) { alert(message); }
    }
}

// Function to format signal cell in the table
function formatSignalCell(cell, signal) {
    cell.textContent = signal;
    cell.className = 'px-6 py-4 whitespace-nowrap text-sm font-semibold'; // Reset classes
    switch (signal) {
        case 'BUY':
            cell.classList.add('text-green-600');
            break;
        case 'SELL':
            cell.classList.add('text-red-600');
            break;
        case 'HOLD':
            cell.classList.add('text-gray-600');
            break;
        default: // Includes 'Error', 'Fetching...', etc.
            cell.classList.add('text-gray-500');
            break;
    }
}

// Function to fetch and display account info
async function fetchAccountInfo() {
    if (buyingPowerSpan) buyingPowerSpan.textContent = 'Fetching...'; 
    try {
        const response = await fetch('/api/account');
        if (!response.ok) {
            let errorDetail = response.statusText;
            try {
                 const errorData = await response.json();
                 errorDetail = errorData.detail || JSON.stringify(errorData);
            } catch (e) { /* Ignore */ }
            console.error(`Error fetching account info: ${response.status}`, errorDetail);
            displayError(`Failed to fetch account info: ${errorDetail}`); // Use general error display
            if (buyingPowerSpan) buyingPowerSpan.textContent = 'Error';
            return; 
        }
        const accountData = await response.json();
        if (buyingPowerSpan) {
             const bp = parseFloat(accountData.buying_power);
             buyingPowerSpan.textContent = isNaN(bp) ? 'Error: Invalid Value' : bp.toFixed(2);
        }
    } catch (error) {
        console.error('Fetch Account Info Network/Script Error:', error);
        displayError('Network error fetching account data. Check console.');
        if (buyingPowerSpan) buyingPowerSpan.textContent = 'Error';
    }
}

// --- Event Listeners ---

fetchBtn.addEventListener('click', async () => {
    displayError('');
    resultsTbody.innerHTML = '';
    const rawInput = symbolsTextarea.value;
    const symbols = rawInput
        .split(/[\n,]+/)
        .map(s => s.toUpperCase().trim())
        .filter(s => s !== '');

    if (symbols.length === 0) {
        displayError('Please enter at least one valid stock symbol.');
        resultsTbody.innerHTML = `<tr><td colspan="5" class="px-6 py-4 text-sm text-gray-500 text-center">No symbols entered.</td></tr>`;
        return;
    }

    // Show loading state
    resultsTbody.innerHTML = `<tr><td colspan="5" class="px-6 py-4 text-sm text-gray-500 text-center">Fetching data for ${symbols.length} symbol(s)...</td></tr>`;

    // Fetch account info and positions first (concurrently)
    let positions = {}; // Default to empty positions
    try {
        const [accountInfoResult, positionsResult] = await Promise.all([
             fetchAccountInfo(), // Still useful for buying power display
             fetch('/api/positions') // Fetch positions data
        ]);

        // Process positions response
        if (positionsResult && positionsResult.ok) {
             positions = await positionsResult.json();
             console.log("Fetched Positions:", positions);
        } else if (positionsResult) {
             console.error(`Error fetching positions: ${positionsResult.status}`, await positionsResult.text());
             displayError('Failed to fetch current positions. Quantity info may be inaccurate.');
             // Continue without position data, defaults to 0
        } else {
             displayError('Failed to fetch current positions. Network error?');
        }
    } catch (error) {
         console.error("Error during initial account/position fetch:", error);
         displayError('Error fetching account/position data. Check console.');
         // Continue without position data
    }
    
    // Create Fetch Promises for Each Symbol (Price & Signal)
    const fetchPromises = symbols.map(async (symbol) => {
        try {
            // Fetch price and signal concurrently for this symbol
            const [priceResponse, signalResponse] = await Promise.all([
                fetch(`/api/price/${symbol}`),
                fetch(`/api/signal/${symbol}`)
            ]);

            let price = 'Error';
            let signal = 'Error';
            let priceError = null;
            let signalError = null;

            // Process price response
            if (!priceResponse.ok) {
                priceError = priceResponse.statusText;
                try { priceError = (await priceResponse.json()).detail || priceError; } catch (e) {} 
                console.error(`Price fetch error for ${symbol}: ${priceResponse.status}`, priceError);
            } else {
                try {
                    const priceData = await priceResponse.json();
                    const p = parseFloat(priceData.price);
                    price = isNaN(p) ? 'Error' : p.toFixed(2);
                } catch (e) { priceError = 'Parsing error'; console.error(`Price parsing error for ${symbol}:`, e); }
            }

            // Process signal response
            if (!signalResponse.ok) {
                signalError = signalResponse.statusText;
                try { signalError = (await signalResponse.json()).detail || signalError; } catch (e) {} 
                console.error(`Signal fetch error for ${symbol}: ${signalResponse.status}`, signalError);
            } else {
                 try {
                     const signalData = await signalResponse.json();
                     signal = signalData.signal || 'Error';
                 } catch (e) { signalError = 'Parsing error'; console.error(`Signal parsing error for ${symbol}:`, e); }
            }
            
            // Return structured result for this symbol
            return { symbol, price, signal, error: priceError || signalError };

        } catch (networkError) {
            console.error(`Network/Script error fetching data for ${symbol}:`, networkError);
            return { symbol, price: 'Error', signal: 'Error', error: 'Network error' };
        }
    });

    // Wait for All Price/Signal Fetches and Populate Table
    const results = await Promise.allSettled(fetchPromises);
    resultsTbody.innerHTML = ''; // Clear loading message

    results.forEach(result => {
        const data = result.status === 'fulfilled' ? result.value : { symbol: 'Unknown', price: 'Error', signal: 'Error', error: 'Fetch promise rejected' };
        const symbol = data.symbol;
        const currentQty = positions[symbol] || 0; // Get qty from fetched positions, default 0

        const tr = document.createElement('tr');

        // Symbol, Price, Signal cells (existing logic)
        const symbolTd = document.createElement('td');
        symbolTd.className = 'px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900';
        symbolTd.textContent = symbol;
        tr.appendChild(symbolTd);

        const priceTd = document.createElement('td');
        priceTd.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
        priceTd.textContent = data.price === 'Error' ? 'Error' : `$${data.price}`;
        if (data.price === 'Error') priceTd.classList.add('text-red-500');
        tr.appendChild(priceTd);

        const signalTd = document.createElement('td');
        formatSignalCell(signalTd, data.signal); 
        tr.appendChild(signalTd);

        // Position Qty Cell
        const positionTd = document.createElement('td');
        positionTd.className = 'px-6 py-4 whitespace-nowrap text-sm text-gray-500';
        positionTd.textContent = currentQty === 'Error' ? 'Error' : currentQty;
        if (currentQty === 'Error') positionTd.classList.add('text-red-500');
        tr.appendChild(positionTd);

        // Actions Cell
        const actionTd = document.createElement('td');
        actionTd.className = 'px-6 py-4 whitespace-nowrap text-sm font-medium';
        
        // Add BUY button only if position is 0
        if (currentQty === 0) { 
            const buyButton = document.createElement('button');
            buyButton.textContent = 'Buy';
            buyButton.className = 'bg-green-500 text-white px-2 py-1 rounded text-xs hover:bg-green-600 disabled:opacity-50';
            buyButton.dataset.symbol = symbol;
            buyButton.dataset.signal = 'BUY'; // Explicitly set action to BUY
            buyButton.addEventListener('click', handleTradeAction);
            actionTd.appendChild(buyButton);
        } else {
            actionTd.textContent = '-'; // Placeholder if position exists
            // Could add a SELL button here later based on position.qty > 0
        }
        tr.appendChild(actionTd);

        resultsTbody.appendChild(tr);

        if (data.error && !errorMessageDiv.textContent) {
             displayError(`Error fetching data for ${data.symbol}: ${data.error}`);
        }
    });
    
    if (resultsTbody.innerHTML === '') { 
         resultsTbody.innerHTML = `<tr><td colspan="5" class="px-6 py-4 text-sm text-gray-500 text-center">Failed to fetch data for all symbols.</td></tr>`;
    }
});

// Function to handle trade execution from action buttons
async function handleTradeAction(event) {
    displayError(''); // Clear previous errors
    const button = event.target;
    const symbol = button.dataset.symbol;
    const signal = button.dataset.signal; // Should be 'BUY' in this case
    const qty = DEFAULT_QTY; // Use default quantity for now

    if (!symbol || !signal || signal !== 'BUY') { // Basic validation
        console.error('Invalid trade action data:', { symbol, signal });
        displayError('Cannot execute trade: Invalid action data.');
        return;
    }

    button.disabled = true;
    button.textContent = 'Buying...';

    const requestBody = { symbol, signal, qty };

    try {
        const response = await fetch('/api/execute_trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        if (response.ok) {
            const responseData = await response.json();
            console.log('Manual Buy Success:', responseData);
            alert(`Manual Buy Submitted Successfully!\nSymbol: ${responseData.symbol}\nOrder ID: ${responseData.order_id}\nStatus: ${responseData.order_status}`);
            // Refresh data after successful trade?
            // Maybe disable button permanently or remove row/update position?
            button.textContent = 'Bought'; 
            // Consider calling fetchAccountInfo() and potentially refreshing the row data
        } else {
            let errorDetail = response.statusText;
            try { errorDetail = (await response.json()).detail || errorDetail; } catch (e) {} 
            console.error(`Manual Buy Failed: ${response.status}`, errorDetail);
            displayError(`Manual buy failed for ${symbol}: ${errorDetail}`);
            button.disabled = false; // Re-enable on failure
            button.textContent = 'Buy';
        }
    } catch (error) {
        console.error('Manual Buy Network/Script Error:', error);
        displayError(`Network error submitting manual buy for ${symbol}. Check console.`);
        button.disabled = false; // Re-enable on failure
        button.textContent = 'Buy';
    }
}

// Remove old executeBtn listener
// executeBtn.addEventListener('click', async () => { ... });
