// OptiForge Frontend Application

class OptiForgeApp {
    constructor() {
        this.variableCount = 0;
        this.constraintCount = 0;
        this.init();
    }

    init() {
        this.bindEvents();
        this.addVariable(); // Start with one variable
        this.addConstraint(); // Start with one constraint
    }

    bindEvents() {
        document.getElementById('add-variable').addEventListener('click', () => this.addVariable());
        document.getElementById('add-constraint').addEventListener('click', () => this.addConstraint());
        document.getElementById('solve-btn').addEventListener('click', () => this.solve());
        document.getElementById('clear-btn').addEventListener('click', () => this.clear());
    }

    addVariable() {
        this.variableCount++;
        const container = document.getElementById('variables-container');
        const row = document.createElement('div');
        row.className = 'variable-row';
        row.innerHTML = `
            <input type="text" class="var-name" placeholder="Name (e.g., x${this.variableCount})" value="x${this.variableCount}">
            <input type="number" class="var-coeff" placeholder="Obj. Coeff." step="any" value="1">
            <select class="var-type">
                <option value="continuous">Continuous</option>
                <option value="integer">Integer</option>
                <option value="binary">Binary</option>
            </select>
            <input type="number" class="var-lb" placeholder="Lower Bound" step="any" value="0">
            <input type="number" class="var-ub" placeholder="Upper Bound" step="any">
            <button type="button" class="btn btn-danger btn-sm remove-var" title="Remove variable">×</button>
        `;
        container.appendChild(row);
        row.querySelector('.remove-var').addEventListener('click', () => {
            row.remove();
            this.updateConstraintVariableSelects();
        });
        // Update constraint variable selects when a new variable is added
        this.updateConstraintVariableSelects();
    }

    addConstraint() {
        this.constraintCount++;
        const container = document.getElementById('constraints-container');
        const row = document.createElement('div');
        row.className = 'constraint-row';
        row.innerHTML = `
            <input type="text" class="constr-name" placeholder="Name (e.g., c${this.constraintCount})" value="c${this.constraintCount}">
            <div class="constraint-coeffs" data-constraint="${this.constraintCount}">
                <div class="coeff-pair">
                    <select class="coeff-var"></select>
                    <input type="number" class="coeff-val" placeholder="Coeff." step="any" value="1">
                    <button type="button" class="btn btn-danger btn-sm remove-coeff" title="Remove coefficient" style="display: none;">×</button>
                </div>
            </div>
            <button type="button" class="btn btn-secondary btn-sm add-coeff" title="Add coefficient">+ Coeff</button>
            <select class="constr-sense">
                <option value="<=">&le;</option>
                <option value=">=">&ge;</option>
                <option value="==">=</option>
            </select>
            <input type="number" class="constr-rhs" placeholder="RHS" step="any" value="1">
            <button type="button" class="btn btn-danger btn-sm remove-constr" title="Remove constraint">×</button>
        `;
        container.appendChild(row);
        row.querySelector('.remove-constr').addEventListener('click', () => row.remove());
        
        // Add event listener for adding coefficients
        row.querySelector('.add-coeff').addEventListener('click', () => this.addCoefficient(row));
        
        // Add event listener for removing coefficients (event delegation)
        row.querySelector('.constraint-coeffs').addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-coeff')) {
                this.removeCoefficient(e.target);
            }
        });
        
        this.updateConstraintVariableSelects();
    }

    addCoefficient(constraintRow) {
        const coeffsContainer = constraintRow.querySelector('.constraint-coeffs');
        const pair = document.createElement('div');
        pair.className = 'coeff-pair';
        pair.innerHTML = `
            <select class="coeff-var"></select>
            <input type="number" class="coeff-val" placeholder="Coeff." step="any" value="1">
            <button type="button" class="btn btn-danger btn-sm remove-coeff" title="Remove coefficient">×</button>
        `;
        coeffsContainer.appendChild(pair);
        
        // Update the variable selects for the new coefficient dropdown
        this.updateConstraintVariableSelects();
        
        // Show/hide remove buttons based on number of coefficients
        this.updateRemoveCoeffButtons(constraintRow);
    }

    removeCoefficient(button) {
        const pair = button.closest('.coeff-pair');
        const constraintRow = button.closest('.constraint-row');
        pair.remove();
        this.updateRemoveCoeffButtons(constraintRow);
    }

    updateRemoveCoeffButtons(constraintRow) {
        const pairs = constraintRow.querySelectorAll('.coeff-pair');
        pairs.forEach((pair, index) => {
            const removeBtn = pair.querySelector('.remove-coeff');
            if (pairs.length > 1) {
                removeBtn.style.display = 'inline-block';
            } else {
                removeBtn.style.display = 'none';
            }
        });
    }

    updateConstraintVariableSelects() {
        const varNames = Array.from(document.querySelectorAll('.var-name'))
            .map(input => input.value.trim())
            .filter(name => name);

        document.querySelectorAll('.coeff-var').forEach(select => {
            const currentValue = select.value;
            select.innerHTML = varNames.map(name => `<option value="${name}">${name}</option>`).join('');
            if (varNames.includes(currentValue)) {
                select.value = currentValue;
            }
        });
    }

    collectFormData() {
        // Collect variables
        const variables = [];
        document.querySelectorAll('.variable-row').forEach(row => {
            const name = row.querySelector('.var-name').value.trim();
            const coeff = parseFloat(row.querySelector('.var-coeff').value) || 0;
            const type = row.querySelector('.var-type').value;
            const lb = parseFloat(row.querySelector('.var-lb').value) || 0;
            const ubInput = row.querySelector('.var-ub').value.trim();
            const ub = ubInput ? parseFloat(ubInput) : null;

            if (!name) return;

            variables.push({
                name,
                lower_bound: lb,
                upper_bound: ub,
                variable_type: type
            });
        });

        // Collect constraints
        const constraints = [];
        document.querySelectorAll('.constraint-row').forEach(row => {
            const name = row.querySelector('.constr-name').value.trim();
            const sense = row.querySelector('.constr-sense').value;
            const rhs = parseFloat(row.querySelector('.constr-rhs').value) || 0;

            if (!name) return;

            const coeffs = {};
            row.querySelectorAll('.coeff-pair').forEach(pair => {
                const varName = pair.querySelector('.coeff-var').value;
                const coeff = parseFloat(pair.querySelector('.coeff-val').value) || 0;
                if (varName && coeff !== 0) {
                    coeffs[varName] = coeff;
                }
            });

            if (Object.keys(coeffs).length === 0) return;

            constraints.push({
                name,
                coefficients: coeffs,
                sense,
                rhs
            });
        });

        // Collect objective
        const objectiveCoeffs = {};
        document.querySelectorAll('.variable-row').forEach(row => {
            const name = row.querySelector('.var-name').value.trim();
            const coeff = parseFloat(row.querySelector('.var-coeff').value) || 0;
            if (name && coeff !== 0) {
                objectiveCoeffs[name] = coeff;
            }
        });

        const objectiveSense = document.getElementById('objective-sense').value;

        return {
            variables,
            constraints,
            objective: {
                coefficients: objectiveCoeffs,
                sense: objectiveSense
            }
        };
    }

    async solve() {
        const solveBtn = document.getElementById('solve-btn');
        const btnText = solveBtn.querySelector('.btn-text');
        const btnLoading = solveBtn.querySelector('.btn-loading');

        // Show loading state
        solveBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoading.style.display = 'inline';

        // Collect form data
        const requestData = this.collectFormData();

        // Validate
        if (requestData.variables.length === 0) {
            this.showError('At least one variable is required');
            this.resetButton(solveBtn, btnText, btnLoading);
            return;
        }

        if (Object.keys(requestData.objective.coefficients).length === 0) {
            this.showError('At least one objective coefficient must be non-zero');
            this.resetButton(solveBtn, btnText, btnLoading);
            return;
        }

        try {
            const response = await fetch('/solve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });

            const data = await response.json();

            if (!response.ok) {
                // Create an error with status code attached
                const error = new Error(data.detail || data.error || 'Unknown error');
                error.statusCode = response.status;
                throw error;
            }

            this.displayResults(data);
        } catch (error) {
            this.showError(error.message, error.statusCode);
        } finally {
            this.resetButton(solveBtn, btnText, btnLoading);
        }
    }

    resetButton(solveBtn, btnText, btnLoading) {
        solveBtn.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }

    displayResults(data) {
        const container = document.getElementById('results-content');

        const statusClass = `status-${data.status}`;
        const statusLabel = data.status.charAt(0).toUpperCase() + data.status.slice(1).replace('_', ' ');

        let html = `
            <div class="result-header">
                <span class="status-badge ${statusClass}">${statusLabel}</span>
                <div class="result-meta">
                    ${data.objective_value !== null ? `<span>Objective: <strong>${data.objective_value.toFixed(6)}</strong></span>` : ''}
                    <span>Iterations: <strong>${data.iterations}</strong></span>
                    ${data.node_count !== null && data.node_count !== undefined ? `<span>Nodes: <strong>${data.node_count}</strong></span>` : ''}
                    ${data.solve_time_ms !== null ? `<span>Time: <strong>${data.solve_time_ms.toFixed(2)} ms</strong></span>` : ''}
                </div>
            </div>
        `;

        // Status explanation
        if (data.status_explanation) {
            html += `<div class="status-explanation">${this.escapeHtml(data.status_explanation)}</div>`;
        }

        // MILP metrics
        if (data.node_count !== null && data.node_count !== undefined) {
            html += `
                <div class="milp-metrics">
                    <h3>MILP Solver Metrics</h3>
                    <div class="metrics-grid">
                        ${data.incumbent_objective !== null && data.incumbent_objective !== undefined ? `<div class="metric"><span class="metric-label">Incumbent Objective</span><span class="metric-value">${data.incumbent_objective.toFixed(6)}</span></div>` : ''}
                        ${data.best_bound !== null && data.best_bound !== undefined ? `<div class="metric"><span class="metric-label">Best Bound</span><span class="metric-value">${data.best_bound.toFixed(6)}</span></div>` : ''}
                        ${data.optimality_gap !== null && data.optimality_gap !== undefined ? `<div class="metric"><span class="metric-label">Optimality Gap</span><span class="metric-value">${(data.optimality_gap * 100).toFixed(2)}%</span></div>` : ''}
                        ${data.absolute_gap !== null && data.absolute_gap !== undefined ? `<div class="metric"><span class="metric-label">Absolute Gap</span><span class="metric-value">${data.absolute_gap.toFixed(6)}</span></div>` : ''}
                        ${data.nodes_remaining !== null && data.nodes_remaining !== undefined ? `<div class="metric"><span class="metric-label">Nodes Remaining</span><span class="metric-value">${data.nodes_remaining}</span></div>` : ''}
                        ${data.max_depth !== null && data.max_depth !== undefined ? `<div class="metric"><span class="metric-label">Max Tree Depth</span><span class="metric-value">${data.max_depth}</span></div>` : ''}
                        ${data.feasible_solutions_found !== null && data.feasible_solutions_found !== undefined ? `<div class="metric"><span class="metric-label">Feasible Solutions Found</span><span class="metric-value">${data.feasible_solutions_found}</span></div>` : ''}
                    </div>
                </div>
            `;
        }

        if (data.message) {
            html += `<div class="error-message">${this.escapeHtml(data.message)}</div>`;
        }

        // Feasibility info
        if (data.feasibility_info) {
            const feasibilityClass = data.feasibility_info.startsWith('All constraints satisfied') ? 'feasibility-ok' : 'feasibility-violated';
            html += `<div class="feasibility-info ${feasibilityClass}">${this.escapeHtml(data.feasibility_info)}</div>`;
        }

        if (data.variable_values && data.variable_values.length > 0) {
            html += `
                <table class="variables-table">
                    <thead>
                        <tr>
                            <th>Variable</th>
                            <th>Type</th>
                            <th>Value</th>
                            <th>Lower Bound</th>
                            <th>Upper Bound</th>
                            <th>Integrality Valid</th>
                            <th>Integrality Violation</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            data.variable_values.forEach(v => {
                const integralityValid = v.integrality_valid !== null && v.integrality_valid !== undefined 
                    ? (v.integrality_valid ? '✓ Yes' : '✗ No') 
                    : 'N/A';
                const integralityViolation = v.integrality_violation !== null && v.integrality_violation !== undefined
                    ? v.integrality_violation.toFixed(6)
                    : 'N/A';
                const lowerBound = v.lower_bound !== null && v.lower_bound !== undefined ? v.lower_bound.toFixed(6) : 'N/A';
                const upperBound = v.upper_bound !== null && v.upper_bound !== undefined ? v.upper_bound.toFixed(6) : 'N/A';
                html += `
                    <tr>
                        <td>${this.escapeHtml(v.name)}</td>
                        <td>${v.variable_type}</td>
                        <td>${v.value.toFixed(6)}</td>
                        <td>${lowerBound}</td>
                        <td>${upperBound}</td>
                        <td>${integralityValid}</td>
                        <td>${integralityViolation}</td>
                    </tr>
                `;
            });
            html += `
                    </tbody>
                </table>
            `;
        } else if (data.status !== 'infeasible' && data.status !== 'unbounded') {
            html += '<p class="empty-state">No variable values in solution.</p>';
        }

        // Constraint evaluations table
        if (data.constraint_evaluations && data.constraint_evaluations.length > 0) {
            html += `
                <table class="constraints-table">
                    <thead>
                        <tr>
                            <th>Constraint</th>
                            <th>Sense</th>
                            <th>LHS Value</th>
                            <th>RHS</th>
                            <th>Slack</th>
                            <th>Residual</th>
                            <th>Violation</th>
                            <th>Satisfied</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            data.constraint_evaluations.forEach(ce => {
                const slack = ce.slack !== null && ce.slack !== undefined ? ce.slack.toFixed(6) : 'N/A';
                const residual = ce.residual !== null && ce.residual !== undefined ? ce.residual.toFixed(6) : 'N/A';
                const violation = ce.violation !== null && ce.violation !== undefined ? ce.violation.toFixed(6) : 'N/A';
                const satisfied = ce.is_satisfied ? '✓ Yes' : '✗ No';
                const rowClass = ce.is_satisfied ? '' : 'constraint-violated';
                html += `
                    <tr class="${rowClass}">
                        <td>${this.escapeHtml(ce.name)}</td>
                        <td>${ce.sense}</td>
                        <td>${ce.lhs_value.toFixed(6)}</td>
                        <td>${ce.rhs.toFixed(6)}</td>
                        <td>${slack}</td>
                        <td>${residual}</td>
                        <td>${violation}</td>
                        <td>${satisfied}</td>
                    </tr>
                `;
            });
            html += `
                    </tbody>
                </table>
            `;
        }

        container.innerHTML = html;
    }

    showError(message, statusCode = null) {
        const container = document.getElementById('results-content');
        
        // Categorize error based on status code and message
        const errorInfo = this.categorizeError(message, statusCode);
        
        let html = `
            <div class="error-message error-${errorInfo.category}">
                <div class="error-header">
                    <span class="error-icon">${errorInfo.icon}</span>
                    <strong class="error-title">${errorInfo.title}</strong>
                </div>
                <div class="error-description">${this.escapeHtml(errorInfo.description)}</div>
                ${errorInfo.suggestion ? `<div class="error-suggestion">${this.escapeHtml(errorInfo.suggestion)}</div>` : ''}
                ${errorInfo.retryable ? `
                    <div class="error-actions">
                        <button type="button" class="btn btn-primary retry-btn" onclick="window.app.retrySolve()">
                            Try Again
                        </button>
                    </div>
                ` : ''}
            </div>
        `;
        
        container.innerHTML = html;
    }

    categorizeError(message, statusCode) {
        const msg = message.toLowerCase();
        
        // Timeout errors
        if (statusCode === 408 || msg.includes('timeout') || msg.includes('timed out')) {
            return {
                category: 'timeout',
                icon: '⏱️',
                title: 'Solver Timeout',
                description: 'The solver took too long to find a solution.',
                suggestion: 'Try simplifying the problem, reducing the number of variables/constraints, or increasing the timeout.',
                retryable: true
            };
        }
        
        // Request too large
        if (statusCode === 413 || msg.includes('too large') || msg.includes('request body')) {
            return {
                category: 'size',
                icon: '📦',
                title: 'Request Too Large',
                description: 'The problem definition exceeds the maximum allowed size.',
                suggestion: 'Reduce the number of variables or constraints.',
                retryable: false
            };
        }
        
        // Validation errors (400)
        if (statusCode === 400 || msg.includes('validation') || msg.includes('invalid') || 
            msg.includes('must be') || msg.includes('required') || msg.includes('duplicate')) {
            return {
                category: 'validation',
                icon: '⚠️',
                title: 'Invalid Input',
                description: message,
                suggestion: 'Check your variable names, bounds, constraint coefficients, and objective function.',
                retryable: false
            };
        }
        
        // Infeasible
        if (msg.includes('infeasible')) {
            return {
                category: 'infeasible',
                icon: '🚫',
                title: 'Problem Infeasible',
                description: 'No solution exists that satisfies all constraints.',
                suggestion: 'Check your constraints for conflicts. Try relaxing bounds or removing constraints.',
                retryable: false
            };
        }
        
        // Unbounded
        if (msg.includes('unbounded')) {
            return {
                category: 'unbounded',
                icon: '📈',
                title: 'Problem Unbounded',
                description: 'The objective can be improved indefinitely.',
                suggestion: 'Add upper bounds to variables or add constraints to limit the feasible region.',
                retryable: false
            };
        }
        
        // Iteration/Node limit
        if (msg.includes('iteration limit') || msg.includes('node limit')) {
            return {
                category: 'limit',
                icon: '🔄',
                title: 'Solver Limit Reached',
                description: message,
                suggestion: 'The problem may be too complex. Try simplifying or increasing limits if available.',
                retryable: true
            };
        }
        
        // Numerical errors
        if (msg.includes('singular') || msg.includes('numerical') || msg.includes('condition')) {
            return {
                category: 'numerical',
                icon: '🔢',
                title: 'Numerical Issue',
                description: 'The solver encountered numerical difficulties.',
                suggestion: 'Try scaling your problem (avoid very large or very small coefficients).',
                retryable: true
            };
        }
        
        // Server errors (500)
        if (statusCode === 500 || msg.includes('internal') || msg.includes('server')) {
            return {
                category: 'server',
                icon: '🔧',
                title: 'Server Error',
                description: 'An unexpected error occurred on the server.',
                suggestion: 'Please try again. If the problem persists, contact support.',
                retryable: true
            };
        }
        
        // Generic error
        return {
            category: 'unknown',
            icon: '❌',
            title: 'Error',
            description: message,
            suggestion: 'Please check your input and try again.',
            retryable: true
        };
    }

    retrySolve() {
        this.solve();
    }

    clear() {
        // Clear variables (keep one)
        const varContainer = document.getElementById('variables-container');
        varContainer.innerHTML = '';
        this.variableCount = 0;
        this.addVariable();

        // Clear constraints (keep one)
        const constrContainer = document.getElementById('constraints-container');
        constrContainer.innerHTML = '';
        this.constraintCount = 0;
        this.addConstraint();

        // Reset objective sense
        document.getElementById('objective-sense').value = 'minimize';

        // Clear results
        document.getElementById('results-content').innerHTML = `
            <div class="empty-state">
                <p>Enter a problem and click Solve to see results.</p>
            </div>
        `;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new OptiForgeApp();
});