import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type CustomerStatus = 'active' | 'suspended' | 'offboarded'
type CustomerPlan = 'starter' | 'growth' | 'enterprise'

type CustomerRow = {
  customer_id: string
  display_name: string
  status: CustomerStatus
  plan: CustomerPlan
  region: string
  retention_days: number
  is_demo: boolean
}

type WorkspaceRow = {
  workspace_id: string
  customer_id: string
  display_name: string
  environment: string
  status: string
  monthly_request_quota: number | null
  created_at: string
}

type UsageAggregate = {
  customer_id: string
  workspace_id: string
  date: string
  total_requests: number
  total_tokens_in: number
  total_tokens_out: number
  total_context_tokens: number
  total_vector_reads: number
  total_memory_writes: number
  active_end_users: number
  avg_latency_ms: number
}

type RecentCustomerActivity = {
  customer_id: string
  inbound_calls: number
  outbound_calls: number
  last_seen_at: string
}

type DatabaseSummary = {
  generated_at: string
  recent_window_seconds: number
  customers_total: number
  workspaces_total: number
  credentials_total: number
  active_credentials_total: number
  memory_records_total: number
  active_memory_records_total: number
  usage_events_total: number
  audn_decisions_total: number
  admin_audit_entries_total: number
  active_customers_last_window: number
  active_workspaces_last_window: number
  active_end_users_last_window: number
  mcp_requests_last_window: number
  admin_requests_last_window: number
  last_memory_write_at: string | null
  last_usage_event_at: string | null
}

const defaultCreateForm = {
  display_name: 'New Customer',
  plan: 'starter' as CustomerPlan,
  region: 'us-east-1',
  retention_days: 90,
  is_demo: false,
}

const defaultEditForm = {
  display_name: '',
  status: 'active' as CustomerStatus,
  plan: 'starter' as CustomerPlan,
  region: 'us-east-1',
  retention_days: 90,
  is_demo: false,
}

function App() {
  const apiBase = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'
  const [activeTab, setActiveTab] = useState<'overview' | 'setup' | 'dashboard' | 'activity'>(() => {
    const saved = window.localStorage.getItem('genmind.activeTab')
    if (saved === 'overview' || saved === 'setup' || saved === 'dashboard' || saved === 'activity') {
      return saved
    }
    return 'overview'
  })

  const tokenCacheRef = useRef<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [logLines, setLogLines] = useState<string[]>([])

  const [customers, setCustomers] = useState<CustomerRow[]>([])
  const [filteredCustomers, setFilteredCustomers] = useState<CustomerRow[]>([])
  const [selectedCustomerId, setSelectedCustomerId] = useState(() => {
    return window.localStorage.getItem('genmind.selectedCustomerId') ?? ''
  })
  const [customerSearch, setCustomerSearch] = useState('')
  const [customerVisibility, setCustomerVisibility] = useState<'all' | 'real' | 'demo'>('real')

  const [createForm, setCreateForm] = useState(defaultCreateForm)
  const [editForm, setEditForm] = useState(defaultEditForm)

  const [workspaces, setWorkspaces] = useState<WorkspaceRow[]>([])
  const [usageRows, setUsageRows] = useState<UsageAggregate[]>([])
  const [allUsageRows, setAllUsageRows] = useState<UsageAggregate[]>([])
  const [databaseSummary, setDatabaseSummary] = useState<DatabaseSummary | null>(null)
  const [recentActivityByCustomer, setRecentActivityByCustomer] = useState<Record<string, RecentCustomerActivity>>({})
  const recentActivityInFlightRef = useRef(false)
  const globalDashboardInFlightRef = useRef(false)
  const databaseSummaryInFlightRef = useRef(false)
  const [connectionInstructions, setConnectionInstructions] = useState('')
  const [copyStatus, setCopyStatus] = useState('')
  const [instructionCustomerId, setInstructionCustomerId] = useState('')

  const selectedCustomer = useMemo(
    () => customers.find((row) => row.customer_id === selectedCustomerId) ?? null,
    [customers, selectedCustomerId],
  )

  const selectCustomer = (customerId: string) => {
    setSelectedCustomerId(customerId)
    setInstructionCustomerId(customerId)
  }

  const totals = useMemo(() => {
    return usageRows.reduce(
      (acc, row) => {
        acc.requests += row.total_requests
        acc.tokens += row.total_tokens_in + row.total_tokens_out + row.total_context_tokens
        return acc
      },
      { requests: 0, tokens: 0 },
    )
  }, [usageRows])

  const globalTotals = useMemo(() => {
    return allUsageRows.reduce(
      (acc, row) => {
        acc.requests += row.total_requests
        acc.tokens += row.total_tokens_in + row.total_tokens_out + row.total_context_tokens
        return acc
      },
      { requests: 0, tokens: 0 },
    )
  }, [allUsageRows])

  const customerRollups = useMemo(() => {
    const usageByCustomer = new Map<
      string,
      {
        requests: number
        tokens: number
        totalLatency: number
        latencyRows: number
        workspaces: Set<string>
        activeEndUsers: number
      }
    >()

    for (const row of allUsageRows) {
      const current =
        usageByCustomer.get(row.customer_id) ??
        {
          requests: 0,
          tokens: 0,
          totalLatency: 0,
          latencyRows: 0,
          workspaces: new Set<string>(),
          activeEndUsers: 0,
        }

      current.requests += row.total_requests
      current.tokens += row.total_tokens_in + row.total_tokens_out + row.total_context_tokens
      current.totalLatency += row.avg_latency_ms
      current.latencyRows += 1
      current.workspaces.add(row.workspace_id)
      current.activeEndUsers += row.active_end_users

      usageByCustomer.set(row.customer_id, current)
    }

    return usageByCustomer
  }, [allUsageRows])

  const selectedRecentActivity = selectedCustomerId ? recentActivityByCustomer[selectedCustomerId] ?? null : null

  const selectedCustomerFlow = useMemo(() => {
    if (!selectedCustomer) return null

    const totalMemoryWrites = usageRows.reduce((sum, row) => sum + row.total_memory_writes, 0)
    const totalVectorReads = usageRows.reduce((sum, row) => sum + row.total_vector_reads, 0)
    const totalContextTokens = usageRows.reduce((sum, row) => sum + row.total_context_tokens, 0)
    const activeEndUsers = usageRows.reduce((sum, row) => sum + row.active_end_users, 0)
    const recentInbound = selectedRecentActivity?.inbound_calls ?? 0
    const recentOutbound = selectedRecentActivity?.outbound_calls ?? 0

    return [
      {
        key: 'ask',
        label: 'User asks',
        value: recentInbound,
        note: 'Recent inbound signals over the last 10 seconds.',
      },
      {
        key: 'stream',
        label: 'MCP stream',
        value: recentInbound + recentOutbound,
        note: `${workspaces.length} workspace${workspaces.length === 1 ? '' : 's'} currently attached.`,
      },
      {
        key: 'memory',
        label: 'Memory writes',
        value: totalMemoryWrites,
        note: 'Persisted AUDN updates recorded for this customer.',
      },
      {
        key: 'retrieval',
        label: 'Hybrid retrieval',
        value: totalVectorReads,
        note: `${totalContextTokens} context tokens shaped into prompts.`,
      },
      {
        key: 'answer',
        label: 'Customer answer',
        value: recentOutbound,
        note: `${activeEndUsers} active end-user bucket${activeEndUsers === 1 ? '' : 's'} in current rollups.`,
      },
    ]
  }, [selectedCustomer, selectedRecentActivity, usageRows, workspaces.length])

  const activeCustomersNow = useMemo(() => {
    return customers
      .map((customer) => {
        const recent = recentActivityByCustomer[customer.customer_id]
        const rollup = customerRollups.get(customer.customer_id)
        return {
          customer,
          recent,
          rollup,
          isActive: Boolean(recent && (recent.inbound_calls > 0 || recent.outbound_calls > 0)),
        }
      })
      .filter((entry) => entry.isActive)
      .sort((left, right) => {
        const leftSignals = (left.recent?.inbound_calls ?? 0) + (left.recent?.outbound_calls ?? 0)
        const rightSignals = (right.recent?.inbound_calls ?? 0) + (right.recent?.outbound_calls ?? 0)
        return rightSignals - leftSignals
      })
      .slice(0, 6)
  }, [customerRollups, customers, recentActivityByCustomer])

  const log = (message: string) => {
    const stamp = new Date().toLocaleTimeString()
    setLogLines((prev) => [`[${stamp}] ${message}`, ...prev].slice(0, 30))
  }

  const ensureDevToken = async (customerId: string, workspaceId = 'ws_console'): Promise<string> => {
    const cacheKey = `${customerId}::${workspaceId}`
    const cached = tokenCacheRef.current[cacheKey]
    if (cached) return cached

    const response = await fetch(`${apiBase}/admin/dev/token`, {
      method: 'POST',
      headers: { 'x-dev-bootstrap': 'allow', 'Content-Type': 'application/json' },
      body: JSON.stringify({
        customer_id: customerId,
        workspace_id: workspaceId,
        end_user_id: 'admin_console',
        session_id: 'admin_console',
        scopes: ['admin:*', 'memory:read', 'memory:write'],
        expires_minutes: 120,
      }),
    })
    if (!response.ok) throw new Error(await response.text())
    const data = await response.json()
    const token = data.access_token as string
    tokenCacheRef.current[cacheKey] = token
    return token
  }

  const authorizedGet = async (path: string, customerIdForToken = 'cust_dev', workspaceIdForToken = 'ws_console') => {
    const authToken = await ensureDevToken(customerIdForToken, workspaceIdForToken)
    const response = await fetch(`${apiBase}${path}`, {
      headers: { Authorization: `Bearer ${authToken}` },
    })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  const authorizedJson = async (
    method: 'POST' | 'PATCH',
    path: string,
    payload: unknown,
    customerIdForToken = 'cust_dev',
    workspaceIdForToken = 'ws_console',
  ) => {
    const authToken = await ensureDevToken(customerIdForToken, workspaceIdForToken)
    const response = await fetch(`${apiBase}${path}`, {
      method,
      headers: {
        Authorization: `Bearer ${authToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    })
    if (!response.ok) throw new Error(await response.text())
    return response.json()
  }

  const loadCustomers = async () => {
    setBusy(true)
    try {
      const rows = (await authorizedGet('/admin/customers')) as CustomerRow[]
      setCustomers(rows)
      if (rows.length && !selectedCustomerId) {
        selectCustomer(rows[0].customer_id)
      }
      log(`Loaded ${rows.length} customers.`)
    } catch (error) {
      log(`Load customers failed: ${(error as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const createCustomer = async () => {
    setBusy(true)
    try {
      const customer = (await authorizedJson('POST', '/admin/customers', {
        ...createForm,
        is_demo: false,
      })) as CustomerRow
      setCreateForm(defaultCreateForm)
      selectCustomer(customer.customer_id)
      log(`Customer created: ${customer.customer_id}`)
      await loadCustomers()
    } catch (error) {
      log(`Create customer failed: ${(error as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const saveCustomerEdits = async () => {
    if (!selectedCustomerId) {
      log('Select a customer first.')
      return
    }
    setBusy(true)
    try {
      const payload = {
        display_name: editForm.display_name,
        status: editForm.status,
        plan: editForm.plan,
        region: editForm.region,
        retention_days: editForm.retention_days,
      }
      await authorizedJson('PATCH', `/admin/customers/${selectedCustomerId}`, payload)
      log(`Customer updated: ${selectedCustomerId}`)
      await loadCustomers()
      await loadSelectedCustomerData(selectedCustomerId)
    } catch (error) {
      log(`Update customer failed: ${(error as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const createWorkspace = async () => {
    if (!selectedCustomerId || !selectedCustomer) {
      log('Select a customer first.')
      return
    }

    setBusy(true)
    try {
      const payload = {
        display_name: `${selectedCustomer.display_name} Workspace ${workspaces.length + 1}`,
        environment: 'prod',
        monthly_request_quota: 100000,
      }
      const workspace = (await authorizedJson(
        'POST',
        `/admin/customers/${selectedCustomerId}/workspaces`,
        payload,
        selectedCustomerId,
      )) as WorkspaceRow
      log(`Workspace created: ${workspace.workspace_id}`)
      await loadSelectedCustomerData(selectedCustomerId)
    } catch (error) {
      log(`Create workspace failed: ${(error as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const loadSelectedCustomerData = async (customerId: string) => {
    setBusy(true)
    try {
      const [usage, workspaceRows] = await Promise.all([
        authorizedGet(`/admin/usage/${customerId}`, customerId) as Promise<UsageAggregate[]>,
        authorizedGet(`/admin/customers/${customerId}/workspaces`, customerId) as Promise<WorkspaceRow[]>,
      ])

      setUsageRows(usage)
      setWorkspaces(workspaceRows)
      log(`Loaded dashboard + workspaces for ${customerId}.`)
    } catch (error) {
      log(`Load customer data failed: ${(error as Error).message}`)
    } finally {
      setBusy(false)
    }
  }

  const loadGlobalDashboard = async () => {
    if (globalDashboardInFlightRef.current) return
    globalDashboardInFlightRef.current = true
    setBusy(true)
    try {
      const rows = (await authorizedGet('/admin/customers')) as CustomerRow[]
      const scopedRows = rows.filter((customer) => {
        if (customerVisibility === 'all') return true
        if (customerVisibility === 'demo') return customer.is_demo
        return !customer.is_demo
      })
      const chunks = await Promise.all(
        scopedRows.map(async (customer) => {
          return (await authorizedGet(`/admin/usage/${customer.customer_id}`, customer.customer_id)) as UsageAggregate[]
        }),
      )
      setAllUsageRows(chunks.flat())
      log(`Global dashboard loaded for ${scopedRows.length} customers.`)
    } catch (error) {
      log(`Global dashboard failed: ${(error as Error).message}`)
    } finally {
      globalDashboardInFlightRef.current = false
      setBusy(false)
    }
  }

  const loadRecentActivity = async () => {
    if (recentActivityInFlightRef.current) return
    recentActivityInFlightRef.current = true
    try {
      const rows = (await authorizedGet('/admin/activity/recent?seconds=10')) as RecentCustomerActivity[]
      const byCustomer = rows.reduce<Record<string, RecentCustomerActivity>>((acc, row) => {
        acc[row.customer_id] = row
        return acc
      }, {})
      setRecentActivityByCustomer(byCustomer)
    } catch (error) {
      log(`Recent activity load failed: ${(error as Error).message}`)
    } finally {
      recentActivityInFlightRef.current = false
    }
  }

  const loadDatabaseSummary = async () => {
    if (databaseSummaryInFlightRef.current) return
    databaseSummaryInFlightRef.current = true
    try {
      const summary = (await authorizedGet('/admin/database/summary?window_seconds=600')) as DatabaseSummary
      setDatabaseSummary(summary)
    } catch (error) {
      log(`Database summary failed: ${(error as Error).message}`)
    } finally {
      databaseSummaryInFlightRef.current = false
    }
  }

  const buildConnectionInstructions = async () => {
    const targetCustomerId = instructionCustomerId || selectedCustomerId
    if (!targetCustomerId) {
      log('Select a customer first.')
      return
    }

    const targetCustomer = customers.find((row) => row.customer_id === targetCustomerId)

    let targetWorkspace: WorkspaceRow | null = null
    try {
      if (targetCustomerId === selectedCustomerId && workspaces.length > 0) {
        targetWorkspace = workspaces[0]
      } else {
        const rows = (await authorizedGet(
          `/admin/customers/${targetCustomerId}/workspaces`,
          targetCustomerId,
        )) as WorkspaceRow[]
        targetWorkspace = rows[0] ?? null
      }
    } catch (error) {
      log(`Workspace lookup failed for ${targetCustomerId}: ${(error as Error).message}`)
      return
    }

    if (!targetWorkspace) {
      try {
        const defaultQuota =
          targetCustomer?.plan === 'enterprise' ? 1000000 : targetCustomer?.plan === 'growth' ? 500000 : 100000

        targetWorkspace = (await authorizedJson(
          'POST',
          `/admin/customers/${targetCustomerId}/workspaces`,
          {
            display_name: `${targetCustomer?.display_name ?? 'Customer'} Production`,
            environment: 'prod',
            monthly_request_quota: defaultQuota,
          },
          targetCustomerId,
        )) as WorkspaceRow

        if (targetCustomerId === selectedCustomerId) {
          await loadSelectedCustomerData(targetCustomerId)
        }
        log(`No workspace existed for ${targetCustomerId}; created ${targetWorkspace.workspace_id} automatically.`)
      } catch (error) {
        log(`Could not create default workspace for ${targetCustomerId}: ${(error as Error).message}`)
        return
      }
    }

    const targetWorkspaceId = targetWorkspace.workspace_id

    const instructions = `# GenMind MCP Integration Instructions (${targetCustomerId})

This guide is platform-agnostic and works with any agent builder that supports MCP or HTTP tool integration.

## Part A: Connect to GenMind service

### A1) Base URL
${apiBase}

### A1.1) Unified Streamable MCP URL
${apiBase}/mcp/stream

### A2) Tenant identifiers
- customer_id: ${targetCustomerId}
- workspace_id: ${targetWorkspaceId}
- maker_id: maker_default
- agent_id: agent_default
- end_user_id: end_user_1
- session_id: session_1

### A3) Customer and workspace profile (prefilled)
- customer_name: ${targetCustomer?.display_name ?? 'N/A'}
- customer_plan: ${targetCustomer?.plan ?? 'N/A'}
- customer_region: ${targetCustomer?.region ?? 'N/A'}
- customer_status: ${targetCustomer?.status ?? 'N/A'}
- workspace_name: ${targetWorkspace.display_name}
- workspace_environment: ${targetWorkspace.environment}
- workspace_status: ${targetWorkspace.status}
- workspace_monthly_quota: ${targetWorkspace.monthly_request_quota ?? 0}

### A4) Get access token (local bootstrap)
curl -X POST '${apiBase}/admin/dev/token' \\
  -H 'x-dev-bootstrap: allow' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "customer_id": "${targetCustomerId}",
    "workspace_id": "${targetWorkspaceId}",
    "end_user_id": "admin_console",
    "session_id": "admin_console",
    "scopes": ["memory:read", "memory:write"],
    "expires_minutes": 60
  }'

### A5) Initialize MCP session
curl -X POST '${apiBase}/mcp/initialization' \\
  -H 'Authorization: Bearer <ACCESS_TOKEN>' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "protocol_version": "2026-01-01",
    "client_name": "customer-integration",
    "client_version": "1.0.0"
  }'

### A6) Write memory
curl -X POST '${apiBase}/mcp/tools/update_memory_state' \\
  -H 'Authorization: Bearer <ACCESS_TOKEN>' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "customer_id": "${targetCustomerId}",
    "workspace_id": "${targetWorkspaceId}",
    "maker_id": "maker_default",
    "agent_id": "agent_default",
    "end_user_id": "end_user_1",
    "session_id": "session_1",
    "user_input": "User prefers concise responses",
    "model_output": "Confirmed. I will stay concise.",
    "metadata": {"source": "customer-integration"}
  }'

### A7) Read context
curl -X POST '${apiBase}/mcp/resources' \\
  -H 'Authorization: Bearer <ACCESS_TOKEN>' \\
  -H 'Content-Type: application/json' \\
  -d '{
    "uri": "genmind://sessions/session_1/context",
    "tenant": {
      "customer_id": "${targetCustomerId}",
      "workspace_id": "${targetWorkspaceId}",
      "end_user_id": "end_user_1",
      "session_id": "session_1"
    },
    "maker_id": "maker_default",
    "agent_id": "agent_default",
    "query": "user preferences",
    "max_tokens": 600
  }'

### A8) Production auth
Replace local bootstrap with issued workspace credentials and production JWT flow.

## Part B: Add GenMind to your agent builder platform

Level mapping for your operating model:
- Level 1: customer_id
- Level 2: maker_id and agent_id
- Level 3: end_user_id

### B1) Prepare platform connector settings
- Use your platform's MCP server connector when available.
- If MCP connector is not available, configure HTTP actions/tools that call GenMind endpoints directly.
- Choose auth method supported by your platform (Bearer token, API key, or OAuth).

### B2) Register GenMind server or actions
1. Open your agent builder platform.
2. Add a new external capability (MCP server, tool, action, or connector).
3. Set base URL to: ${apiBase}
4. Configure authentication headers.
5. Save and enable the integration.

### B3) Map operations in your platform
Map all MCP operations through a single URL:
- POST /mcp/stream

JSON-RPC methods to call on that endpoint:
- initialize
- resources/list
- resources/read
- tools/list
- tools/call (name: update_memory_state)
- Optional async events stream: GET /mcp/events

### B4) Validate end-to-end behavior
1. Trigger a memory write from your agent.
2. Trigger a context read for the same tenant tuple.
3. Confirm the response contains the newly written facts.
4. Verify tenant scoping by changing customer/workspace/user/session IDs.

### B5) Compatibility note
GenMind provides a unified streamable MCP entrypoint:
- /mcp/stream

Direct contract routes are still available for diagnostics and internal tooling.

## Reference checklist
- Keep the tenant tuple explicit in every request.
- Never mix customer/workspace context between calls.
- Start with a non-production workspace and short-lived credentials.
- Promote to production only after write/read validation passes.`

    setConnectionInstructions(instructions)
    setCopyStatus('')
    log(`Connection instructions generated for ${targetCustomerId}.`)
  }

  const copyConnectionInstructions = async () => {
    if (!connectionInstructions.trim()) {
      log('Generate instructions first.')
      return
    }
    try {
      await navigator.clipboard.writeText(connectionInstructions)
      setCopyStatus('Copied to clipboard')
      log('Connection instructions copied.')
    } catch (error) {
      setCopyStatus('Copy failed')
      log(`Copy failed: ${(error as Error).message}`)
    }
  }

  useEffect(() => {
    const term = customerSearch.trim().toLowerCase()
    setFilteredCustomers(
      customers.filter((customer) => {
        const matchesVisibility =
          customerVisibility === 'all' ||
          (customerVisibility === 'demo' && customer.is_demo) ||
          (customerVisibility === 'real' && !customer.is_demo)
        if (!matchesVisibility) return false
        if (!term) return true
        return (
          customer.customer_id.toLowerCase().includes(term) ||
          customer.display_name.toLowerCase().includes(term) ||
          customer.status.toLowerCase().includes(term)
        )
      }),
    )
  }, [customerSearch, customers, customerVisibility])

  useEffect(() => {
    void loadCustomers()
    void loadGlobalDashboard()
    void loadRecentActivity()
  }, [])

  useEffect(() => {
    const isLiveOpsTab = activeTab === 'dashboard' || activeTab === 'activity'
    if (!isLiveOpsTab) return

    const pollGlobalDashboard = () => {
      if (document.visibilityState !== 'visible') return
      void loadGlobalDashboard()
    }

    const pollRecentActivity = () => {
      if (document.visibilityState !== 'visible') return
      void loadRecentActivity()
    }

    const pollDatabaseSummary = () => {
      if (document.visibilityState !== 'visible' || activeTab !== 'dashboard') return
      void loadDatabaseSummary()
    }

    pollGlobalDashboard()
    pollRecentActivity()
    pollDatabaseSummary()
    const recentHandle = window.setInterval(pollRecentActivity, 10000)
    const dashboardHandle = window.setInterval(pollGlobalDashboard, 10000)
    const databaseHandle = window.setInterval(pollDatabaseSummary, 10000)
    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        pollGlobalDashboard()
        pollRecentActivity()
        pollDatabaseSummary()
      }
    }
    document.addEventListener('visibilitychange', onVisibilityChange)

    return () => {
      window.clearInterval(recentHandle)
      window.clearInterval(dashboardHandle)
      window.clearInterval(databaseHandle)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [activeTab, customerVisibility])

  useEffect(() => {
    if (!selectedCustomer) return
    setEditForm({
      display_name: selectedCustomer.display_name,
      status: selectedCustomer.status,
      plan: selectedCustomer.plan,
      region: selectedCustomer.region,
      retention_days: selectedCustomer.retention_days,
      is_demo: selectedCustomer.is_demo,
    })
    void loadSelectedCustomerData(selectedCustomer.customer_id)
  }, [selectedCustomer])

  useEffect(() => {
    setInstructionCustomerId(selectedCustomerId)
  }, [selectedCustomerId])

  useEffect(() => {
    window.localStorage.setItem('genmind.activeTab', activeTab)
  }, [activeTab])

  useEffect(() => {
    if (!selectedCustomerId) {
      window.localStorage.removeItem('genmind.selectedCustomerId')
      return
    }
    window.localStorage.setItem('genmind.selectedCustomerId', selectedCustomerId)
  }, [selectedCustomerId])

  useEffect(() => {
    if (!selectedCustomerId) return
    const exists = customers.some((customer) => customer.customer_id === selectedCustomerId)
    if (!exists) {
      setSelectedCustomerId('')
      setInstructionCustomerId('')
    }
  }, [customers, selectedCustomerId])

  return (
    <div className="shell pro-shell">
      <header className="topbar pro-topbar">
        <div>
          <p className="eyebrow">GenMind Admin Studio</p>
          <h1>Customers, Workspaces, and Memory Control</h1>
          <p className="subtitle">
            Designed after modern SaaS admin consoles: searchable customer table, inline editing panel, scoped workspaces,
            and live usage intelligence.
          </p>
        </div>
        <div className="status-pill" data-busy={busy ? 'true' : 'false'}>
          {busy ? 'Syncing' : 'Live'}
        </div>
      </header>

      <nav className="tabs" aria-label="Main views">
        <button
          className={activeTab === 'overview' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button className={activeTab === 'setup' ? 'tab active' : 'tab'} onClick={() => setActiveTab('setup')}>
          Setup
        </button>
        <button
          className={activeTab === 'dashboard' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('dashboard')}
        >
          Dashboard
        </button>
        <button
          className={activeTab === 'activity' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('activity')}
        >
          Activity
        </button>
      </nav>

      {activeTab === 'overview' ? (
        <>
          <section className="grid three">
            <article className="card panel overview-card">
              <h3>Tenant Scope</h3>
              <p className="hint">Canonical isolation tuple currently active in this console.</p>
              <ul className="tuple-list">
                <li>customer_id: {selectedCustomerId || 'Not selected'}</li>
                <li>workspace_id: {workspaces[0]?.workspace_id ?? 'No workspace yet'}</li>
              </ul>
            </article>

            <article className="card panel overview-card">
              <h3>Footprint</h3>
              <div className="kpis compact">
                <div>
                  <span>Customers</span>
                  <strong>{customers.length}</strong>
                </div>
                <div>
                  <span>Workspaces</span>
                  <strong>{workspaces.length}</strong>
                </div>
                <div>
                  <span>Rows In Global Usage</span>
                  <strong>{allUsageRows.length}</strong>
                </div>
              </div>
            </article>

            <article className="card panel overview-card">
              <h3>Quick Actions</h3>
              <div className="actions-inline single-col">
                <button onClick={() => setActiveTab('setup')}>Open Setup</button>
                <button onClick={() => setActiveTab('dashboard')}>Open Dashboard</button>
              </div>
            </article>
          </section>

          <section className="grid two">
            <article className="card panel table-panel">
              <div className="table-header">
                <h2>Customers</h2>
                <div className="table-controls">
                  <select
                    value={customerVisibility}
                    onChange={(event) => setCustomerVisibility(event.target.value as 'all' | 'real' | 'demo')}
                  >
                    <option value="all">All</option>
                    <option value="real">Real only</option>
                    <option value="demo">Demo only</option>
                  </select>
                  <input
                    className="search"
                    placeholder="Search by name, id, status"
                    value={customerSearch}
                    onChange={(event) => setCustomerSearch(event.target.value)}
                  />
                </div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Customer</th>
                    <th>Status</th>
                    <th>Plan</th>
                    <th>Region</th>
                    <th>Type</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredCustomers.map((customer) => (
                    <tr
                      key={customer.customer_id}
                      onClick={() => selectCustomer(customer.customer_id)}
                      className={customer.customer_id === selectedCustomerId ? 'selected-row' : ''}
                    >
                      <td>
                        <strong>{customer.display_name}</strong>
                        <div className="sub-id">{customer.customer_id}</div>
                      </td>
                      <td>
                        <span className={`status-badge ${customer.status}`}>{customer.status}</span>
                      </td>
                      <td>{customer.plan}</td>
                      <td>{customer.region}</td>
                      <td>{customer.is_demo ? 'demo' : 'real'}</td>
                    </tr>
                  ))}
                  {filteredCustomers.length === 0 ? (
                    <tr>
                      <td colSpan={5}>No matching customers.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </article>

            <article className="card panel workspace-panel">
              <h2>Workspaces For Selected Customer</h2>
              <p className="hint">Workspaces isolate environments and credentials under a single customer.</p>
              <ul className="customers">
                {workspaces.map((workspace) => (
                  <li key={workspace.workspace_id}>
                    <strong>{workspace.display_name}</strong>
                    <span>{workspace.workspace_id}</span>
                    <span>
                      {workspace.environment} | {workspace.status} | quota {workspace.monthly_request_quota ?? 0}
                    </span>
                  </li>
                ))}
                {workspaces.length === 0 ? <li>No workspaces yet.</li> : null}
              </ul>
            </article>
          </section>
        </>
      ) : null}

      {activeTab === 'setup' ? (
        <>

      <section className="grid two pro-main-grid">
        <article className="card panel create-panel">
          <h2>Create Customer</h2>
          <label>
            Name
            <input
              value={createForm.display_name}
              onChange={(event) => setCreateForm((prev) => ({ ...prev, display_name: event.target.value }))}
            />
          </label>
          <div className="row">
            <label>
              Plan
              <select
                value={createForm.plan}
                onChange={(event) =>
                  setCreateForm((prev) => ({ ...prev, plan: event.target.value as CustomerPlan }))
                }
              >
                <option value="starter">starter</option>
                <option value="growth">growth</option>
                <option value="enterprise">enterprise</option>
              </select>
            </label>
            <label>
              Region
              <input
                value={createForm.region}
                onChange={(event) => setCreateForm((prev) => ({ ...prev, region: event.target.value }))}
              />
            </label>
          </div>
          <label>
            Retention Days
            <input
              type="number"
              min={7}
              max={730}
              value={createForm.retention_days}
              onChange={(event) =>
                setCreateForm((prev) => ({ ...prev, retention_days: Number(event.target.value) }))
              }
            />
          </label>
          <div className="actions-inline">
            <button onClick={createCustomer}>Create</button>
            <button onClick={loadCustomers}>Refresh</button>
            <button onClick={loadGlobalDashboard}>Global KPIs</button>
          </div>
        </article>

        <article className="card panel table-panel">
          <div className="table-header">
            <h2>All Customers</h2>
            <div className="table-controls">
              <select
                value={customerVisibility}
                onChange={(event) => setCustomerVisibility(event.target.value as 'all' | 'real' | 'demo')}
              >
                <option value="all">All</option>
                <option value="real">Real only</option>
                <option value="demo">Demo only</option>
              </select>
              <input
                className="search"
                placeholder="Search by name, id, status"
                value={customerSearch}
                onChange={(event) => setCustomerSearch(event.target.value)}
              />
            </div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Customer</th>
                <th>Status</th>
                <th>Plan</th>
                <th>Region</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {filteredCustomers.map((customer) => (
                <tr
                  key={customer.customer_id}
                  onClick={() => selectCustomer(customer.customer_id)}
                  className={customer.customer_id === selectedCustomerId ? 'selected-row' : ''}
                >
                  <td>
                    <strong>{customer.display_name}</strong>
                    <div className="sub-id">{customer.customer_id}</div>
                  </td>
                  <td>
                    <span className={`status-badge ${customer.status}`}>{customer.status}</span>
                  </td>
                  <td>{customer.plan}</td>
                  <td>{customer.region}</td>
                  <td>{customer.is_demo ? 'demo' : 'real'}</td>
                </tr>
              ))}
              {filteredCustomers.length === 0 ? (
                <tr>
                  <td colSpan={5}>No matching customers.</td>
                </tr>
              ) : null}
            </tbody>
          </table>
          <div className="actions-inline two-col">
            <button onClick={loadCustomers}>Refresh Customers</button>
          </div>
          <p className="hint status-note">Select a row above to edit that customer below.</p>
        </article>
      </section>

      <section className="grid two">
        <article className="card panel edit-panel">
          <h2>Edit Customer</h2>
          <p className="hint">
            {selectedCustomerId ? `Editing ${selectedCustomerId}` : 'Select a customer row to edit.'}
          </p>
          <label>
            Name
            <input
              value={editForm.display_name}
              onChange={(event) => setEditForm((prev) => ({ ...prev, display_name: event.target.value }))}
              disabled={!selectedCustomerId}
            />
          </label>
          <div className="row">
            <label>
              Status
              <select
                value={editForm.status}
                onChange={(event) =>
                  setEditForm((prev) => ({ ...prev, status: event.target.value as CustomerStatus }))
                }
                disabled={!selectedCustomerId}
              >
                <option value="active">active</option>
                <option value="suspended">suspended</option>
                <option value="offboarded">offboarded</option>
              </select>
            </label>
            <label>
              Plan
              <select
                value={editForm.plan}
                onChange={(event) =>
                  setEditForm((prev) => ({ ...prev, plan: event.target.value as CustomerPlan }))
                }
                disabled={!selectedCustomerId}
              >
                <option value="starter">starter</option>
                <option value="growth">growth</option>
                <option value="enterprise">enterprise</option>
              </select>
            </label>
          </div>
          <div className="row">
            <label>
              Region
              <input
                value={editForm.region}
                onChange={(event) => setEditForm((prev) => ({ ...prev, region: event.target.value }))}
                disabled={!selectedCustomerId}
              />
            </label>
            <label>
              Retention Days
              <input
                type="number"
                min={7}
                max={730}
                value={editForm.retention_days}
                onChange={(event) =>
                  setEditForm((prev) => ({ ...prev, retention_days: Number(event.target.value) }))
                }
                disabled={!selectedCustomerId}
              />
            </label>
          </div>
          <div className="actions-inline">
            <button onClick={saveCustomerEdits} disabled={!selectedCustomerId}>
              Save Changes
            </button>
            <button onClick={createWorkspace} disabled={!selectedCustomerId}>
              Add Workspace
            </button>
          </div>
        </article>

        <article className="card panel workspace-panel">
          <h2>Workspaces</h2>
          <p className="hint">Linked to the selected customer.</p>
          <ul className="customers">
            {workspaces.map((workspace) => (
              <li key={workspace.workspace_id}>
                <strong>{workspace.display_name}</strong>
                <span>{workspace.workspace_id}</span>
                <span>
                  {workspace.environment} | {workspace.status} | quota {workspace.monthly_request_quota ?? 0}
                </span>
              </li>
            ))}
            {workspaces.length === 0 ? <li>No workspaces yet.</li> : null}
          </ul>
        </article>
      </section>

      <section className="grid two">
        <article className="card panel">
          <h2>Customer Connection Instructions</h2>
          <p className="hint">
            Pick a customer, then generate and copy an exact integration checklist for your implementation team.
          </p>
          <label>
            Select Customer For Instructions
            <select
              value={instructionCustomerId}
              onChange={(event) => selectCustomer(event.target.value)}
            >
              <option value="">Select customer</option>
              {customers.map((customer) => (
                <option key={customer.customer_id} value={customer.customer_id}>
                  {customer.display_name} ({customer.customer_id})
                </option>
              ))}
            </select>
          </label>
          <div className="actions-inline two-col">
            <button onClick={() => void buildConnectionInstructions()} disabled={!instructionCustomerId}>
              Generate Instructions
            </button>
          </div>
          <p className="hint status-note">{copyStatus || 'Generated text can be edited before copying.'}</p>
          <label>
            Instructions
            <textarea
              value={connectionInstructions}
              rows={18}
              onChange={(event) => setConnectionInstructions(event.target.value)}
              placeholder="Select a customer, then click Generate Instructions."
            />
          </label>
          <div className="actions-inline">
            <button onClick={copyConnectionInstructions} disabled={!connectionInstructions.trim()}>
              Copy Instructions
            </button>
          </div>
        </article>
      </section>
        </>
      ) : null}

      {activeTab === 'dashboard' ? (
        <>
          <section className="grid two">
            <article className="card panel">
              <h2>Global Dashboard</h2>
              <div className="kpis">
                <div>
                  <span>All Requests</span>
                  <strong>{globalTotals.requests}</strong>
                </div>
                <div>
                  <span>All Tokens (incl. context)</span>
                  <strong>{globalTotals.tokens}</strong>
                </div>
                <div>
                  <span>Customers</span>
                  <strong>{customers.length}</strong>
                </div>
              </div>
              <div className="actions-inline single-col">
                <button onClick={loadGlobalDashboard}>Refresh Global Aggregates</button>
              </div>
            </article>

            <article className="card panel">
              <h2>Database Snapshot</h2>
              <div className="kpis compact">
                <div>
                  <span>Memory Rows</span>
                  <strong>{databaseSummary?.active_memory_records_total ?? 0}</strong>
                </div>
                <div>
                  <span>MCP Events 10m</span>
                  <strong>{databaseSummary?.mcp_requests_last_window ?? 0}</strong>
                </div>
                <div>
                  <span>Active End Users 10m</span>
                  <strong>{databaseSummary?.active_end_users_last_window ?? 0}</strong>
                </div>
              </div>
              <div className="db-snapshot-grid">
                <div className="db-snapshot-item">
                  <span>Customers</span>
                  <strong>{databaseSummary?.customers_total ?? 0}</strong>
                </div>
                <div className="db-snapshot-item">
                  <span>Workspaces</span>
                  <strong>{databaseSummary?.workspaces_total ?? 0}</strong>
                </div>
                <div className="db-snapshot-item">
                  <span>Credentials</span>
                  <strong>
                    {databaseSummary?.active_credentials_total ?? 0}/{databaseSummary?.credentials_total ?? 0}
                  </strong>
                </div>
                <div className="db-snapshot-item">
                  <span>Usage Events</span>
                  <strong>{databaseSummary?.usage_events_total ?? 0}</strong>
                </div>
                <div className="db-snapshot-item">
                  <span>AUDN Decisions</span>
                  <strong>{databaseSummary?.audn_decisions_total ?? 0}</strong>
                </div>
                <div className="db-snapshot-item">
                  <span>Admin Audit</span>
                  <strong>{databaseSummary?.admin_audit_entries_total ?? 0}</strong>
                </div>
              </div>
              <p className="hint status-note">
                Last memory write:{' '}
                {databaseSummary?.last_memory_write_at
                  ? new Date(databaseSummary.last_memory_write_at).toLocaleTimeString()
                  : 'No memory writes yet'}
              </p>
              <p className="hint status-note">
                Last usage event:{' '}
                {databaseSummary?.last_usage_event_at
                  ? new Date(databaseSummary.last_usage_event_at).toLocaleTimeString()
                  : 'No usage events yet'}
              </p>
            </article>
          </section>

          <section className="grid two">
            <article className="card panel">
              <h2>Customer Filter</h2>
              <p className="hint">Select one customer to drill into full customer details.</p>
              <label>
                Customer
                <select
                  value={selectedCustomerId}
                  onChange={(event) => selectCustomer(event.target.value)}
                >
                  <option value="">Select customer</option>
                  {customers.map((customer) => (
                    <option key={customer.customer_id} value={customer.customer_id}>
                      {customer.display_name} ({customer.customer_id})
                    </option>
                  ))}
                </select>
              </label>
              <div className="kpis compact">
                <div>
                  <span>Selected Customer</span>
                  <strong>{selectedCustomer?.display_name ?? 'None'}</strong>
                </div>
                <div>
                  <span>Selected Workspaces</span>
                  <strong>{workspaces.length}</strong>
                </div>
                <div>
                  <span>Selected Usage Rows</span>
                  <strong>{usageRows.length}</strong>
                </div>
              </div>
              <div className="actions-inline single-col">
                <button
                  onClick={() => selectedCustomer && void loadSelectedCustomerData(selectedCustomer.customer_id)}
                  disabled={!selectedCustomer || busy}
                >
                  Refresh Selected Customer Data
                </button>
              </div>
            </article>
          </section>

          <section className="grid two">
            <article className="card panel table-panel">
              <h2>All Customers Aggregated</h2>
              <table>
                <thead>
                  <tr>
                    <th>Customer</th>
                    <th>Live 10s</th>
                    <th>Requests</th>
                    <th>Tokens</th>
                  </tr>
                </thead>
                <tbody>
                  {customers.map((customer) => {
                    const rollup = customerRollups.get(customer.customer_id)
                    const recent = recentActivityByCustomer[customer.customer_id]
                    const hasIn = (recent?.inbound_calls ?? 0) > 0
                    const hasOut = (recent?.outbound_calls ?? 0) > 0
                    return (
                      <tr
                        key={customer.customer_id}
                        onClick={() => selectCustomer(customer.customer_id)}
                        className={customer.customer_id === selectedCustomerId ? 'selected-row' : ''}
                      >
                        <td>
                          <strong>{customer.display_name}</strong>
                          <div className="sub-id">{customer.customer_id}</div>
                        </td>
                        <td>
                          <div className="activity-icons">
                            <span className={hasIn ? 'activity-pill on' : 'activity-pill off'}>IN</span>
                            <span className={hasOut ? 'activity-pill on' : 'activity-pill off'}>OUT</span>
                          </div>
                        </td>
                        <td>{rollup?.requests ?? 0}</td>
                        <td>{rollup?.tokens ?? 0}</td>
                      </tr>
                    )
                  })}
                  {customers.length === 0 ? (
                    <tr>
                      <td colSpan={4}>No customers found.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </article>

            <article className="card panel">
              <h2>Selected Customer Flow</h2>
              <p className="hint">A compact live view of how the selected customer is currently moving through GenMind.</p>
              {selectedCustomerFlow ? (
                <div className="flow-diagram" role="list" aria-label="Selected customer live flow">
                  {selectedCustomerFlow.map((step, index) => (
                    <div key={step.key} className="flow-step-wrap">
                      <article className="flow-step" role="listitem">
                        <span className="flow-step-index">0{index + 1}</span>
                        <strong>{step.label}</strong>
                        <span className="flow-step-value">{step.value}</span>
                        <p>{step.note}</p>
                      </article>
                      {index < selectedCustomerFlow.length - 1 ? <div className="flow-arrow" aria-hidden="true">→</div> : null}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="hint">Select a customer to show the real-time flow diagram.</p>
              )}
            </article>
          </section>

          <section className="grid two">
            <article className="card panel">
              <h2>Customer Dashboard</h2>
              <p className="hint">Detailed view for the selected customer.</p>
              <div className="kpis">
                <div>
                  <span>Requests</span>
                  <strong>{totals.requests}</strong>
                </div>
                <div>
                  <span>Tokens (incl. context)</span>
                  <strong>{totals.tokens}</strong>
                </div>
                <div>
                  <span>Workspace Count</span>
                  <strong>{workspaces.length}</strong>
                </div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Workspace</th>
                    <th>Requests</th>
                    <th>Latency</th>
                  </tr>
                </thead>
                <tbody>
                  {usageRows.map((row) => (
                    <tr key={`${row.workspace_id}-${row.date}`}>
                      <td>{row.date}</td>
                      <td>{row.workspace_id}</td>
                      <td>{row.total_requests}</td>
                      <td>{row.avg_latency_ms.toFixed(1)} ms</td>
                    </tr>
                  ))}
                  {usageRows.length === 0 ? (
                    <tr>
                      <td colSpan={4}>No usage data for selected customer.</td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </article>
          </section>
        </>
      ) : null}

      {activeTab === 'activity' ? (
        <>
          <section className="grid two">
            <article className="card panel overview-card">
              <h2>Live Pulse</h2>
              <div className="kpis compact">
                <div>
                  <span>Customers Active Now</span>
                  <strong>{activeCustomersNow.length}</strong>
                </div>
                <div>
                  <span>Selected Customer Live</span>
                  <strong>{selectedRecentActivity ? 'Yes' : 'No'}</strong>
                </div>
                <div>
                  <span>Signals Window</span>
                  <strong>10s</strong>
                </div>
              </div>
              <p className="hint">This view highlights only the most recent customer-service movement and the most useful admin notes.</p>
            </article>

            <article className="card panel overview-card">
              <h2>Selected Customer Pulse</h2>
              {selectedCustomer ? (
                <div className="activity-summary">
                  <div>
                    <strong>{selectedCustomer.display_name}</strong>
                    <div className="sub-id">{selectedCustomer.customer_id}</div>
                  </div>
                  <div className="activity-icons">
                    <span className={selectedRecentActivity?.inbound_calls ? 'activity-pill on' : 'activity-pill off'}>IN</span>
                    <span className={selectedRecentActivity?.outbound_calls ? 'activity-pill on' : 'activity-pill off'}>OUT</span>
                  </div>
                  <div className="activity-meta-grid">
                    <span>Requests: {customerRollups.get(selectedCustomer.customer_id)?.requests ?? 0}</span>
                    <span>Tokens: {customerRollups.get(selectedCustomer.customer_id)?.tokens ?? 0}</span>
                    <span>Workspaces: {customerRollups.get(selectedCustomer.customer_id)?.workspaces.size ?? 0}</span>
                    <span>Last Seen: {selectedRecentActivity ? new Date(selectedRecentActivity.last_seen_at).toLocaleTimeString() : 'No recent traffic'}</span>
                  </div>
                </div>
              ) : (
                <p className="hint">Select a customer to see their current traffic pulse.</p>
              )}
            </article>
          </section>

          <section className="grid two">
            <article className="card panel table-panel">
              <h2>Active Customers Now</h2>
              <ul className="customers activity-list">
                {activeCustomersNow.map(({ customer, recent, rollup }) => (
                  <li key={customer.customer_id} onClick={() => selectCustomer(customer.customer_id)}>
                    <strong>{customer.display_name}</strong>
                    <span>{customer.customer_id}</span>
                    <span>
                      IN {recent?.inbound_calls ?? 0} | OUT {recent?.outbound_calls ?? 0} | req {rollup?.requests ?? 0} | tok {rollup?.tokens ?? 0}
                    </span>
                  </li>
                ))}
                {activeCustomersNow.length === 0 ? <li>No customers have recent MCP traffic.</li> : null}
              </ul>
            </article>

            <article className="card panel">
              <h2>Recent Notes</h2>
              <ul className="log compact-log">
                {logLines.slice(0, 8).map((line, index) => (
                  <li key={`${index}-${line}`}>{line}</li>
                ))}
                {logLines.length === 0 ? <li>No actions yet.</li> : null}
              </ul>
            </article>
          </section>
        </>
      ) : null}

    </div>
  )
}

export default App
