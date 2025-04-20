"use client"

import { CardTitle } from "@/components/ui/card"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Keyboard, X, Send, Check, Copy, Terminal, AlertTriangle, Zap, Clock, FileCode } from "lucide-react"
import { cn } from "@/lib/utils"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Slider } from "@/components/ui/slider"
import { FixTemplatesDialog } from "@/components/fix-templates-dialog"
import type { FixTemplate } from "@/components/fix-templates-library"
import { toast } from "@/components/ui/toast"
import { generateText } from "ai"
import { openai } from "@ai-sdk/openai"

interface Message {
  id: number
  content: string
  sender: "user" | "agent" | "system"
  timestamp: Date
  type?: "error" | "fix" | "info" | "success" | "warning" | "auto-fix" | "template"
  codeSnippet?: string
  actions?: Array<{
    label: string
    action: "approve" | "reject" | "edit" | "apply"
    applied?: boolean
  }>
  autoFixed?: boolean
  templateId?: string
  critical?: boolean
}

interface Error {
  id: number
  message: string
  timestamp: Date
  severity: "low" | "medium" | "high"
  autoFixable: boolean
  autoFixed?: boolean
  fixApplied?: boolean
  critical?: boolean
}

interface AutoFixSettings {
  enabled: boolean
  confidenceThreshold: number
  severityLevels: {
    low: boolean
    medium: boolean
    high: boolean
  }
  notifyBeforeFix: boolean
}

export function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [input, setInput] = useState("")
  const [activeTab, setActiveTab] = useState("chat")
  const [messages, setMessages] = useState<Message[]>([])
  const [notifications, setNotifications] = useState(2)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [recentErrors, setRecentErrors] = useState<Error[]>([
    {
      id: 1,
      message: "Schema mismatch in 'property_value'",
      timestamp: new Date(Date.now() - 3600000),
      severity: "high",
      autoFixable: false,
    },
    {
      id: 2,
      message: "Missing values in 'claim_date'",
      timestamp: new Date(Date.now() - 7200000),
      severity: "medium",
      autoFixable: true,
    },
    {
      id: 3,
      message: "Drift detected in 'location_code'",
      timestamp: new Date(Date.now() - 10800000),
      severity: "low",
      autoFixable: true,
    },
  ])

  // Auto-fix settings
  const [autoFixSettings, setAutoFixSettings] = useState<AutoFixSettings>({
    enabled: true,
    confidenceThreshold: 80,
    severityLevels: {
      low: true,
      medium: true,
      high: false,
    },
    notifyBeforeFix: true,
  })

  // Applied fixes history
  const [appliedFixes, setAppliedFixes] = useState<
    Array<{
      id: number
      error: string
      fix: string
      timestamp: Date
      automatic: boolean
      templateId?: string
      critical?: boolean
    }>
  >([
    {
      id: 1,
      error: "Missing values in 'policy_start_date'",
      fix: "Added default value of current date for missing entries",
      timestamp: new Date(Date.now() - 86400000), // 1 day ago
      automatic: true,
    },
    {
      id: 2,
      error: "Incorrect data type in 'zip_code'",
      fix: "Converted numeric zip codes to string format with leading zeros",
      timestamp: new Date(Date.now() - 172800000), // 2 days ago
      automatic: false,
    },
  ])

  const handleSend = async () => {
    if (!input.trim()) return

    // Add user message
    const userMessage: Message = {
      id: messages.length + 1,
      content: input,
      sender: "user",
      timestamp: new Date(),
    }

    setMessages([...messages, userMessage])

    // Process user input and generate response
    try {
      const { text } = await generateText({
        model: openai("gpt-4o"),
        prompt: input,
      })

      const agentMessage: Message = {
        id: messages.length + 2,
        content: text,
        sender: "agent",
        timestamp: new Date(),
        type: "info",
      }
      setMessages((prev) => [...prev, agentMessage])
    } catch (error) {
      console.error("Error generating text:", error)
      toast.error("Failed to generate response from OpenAI")
    }
  }

  const handleActionClick = (messageId: number, actionIndex: number) => {
    setMessages((prev) =>
      prev.map((message) => {
        if (message.id === messageId && message.actions) {
          const updatedActions = [...message.actions]
          updatedActions[actionIndex] = {
            ...updatedActions[actionIndex],
            applied: true,
          }

          return {
            ...message,
            actions: updatedActions,
          }
        }
        return message
      }),
    )

    // Add a confirmation message
    const action = messages.find((m) => m.id === messageId)?.actions?.[actionIndex]
    const originalMessage = messages.find((m) => m.id === messageId)

    if (action && originalMessage) {
      setTimeout(() => {
        const confirmationMessage: Message = {
          id: messages.length + 1,
          content: `Fix ${action.action === "approve" ? "approved" : action.action === "reject" ? "rejected" : "applied"} successfully.`,
          sender: "agent",
          timestamp: new Date(),
          type: "success",
        }
        setMessages((prev) => [...prev, confirmationMessage])

        // Add to applied fixes if it was an approval or apply action
        if (action.action === "approve" || action.action === "apply") {
          setAppliedFixes((prev) => [
            {
              id: prev.length + 1,
              error: originalMessage.content.includes("issue with")
                ? originalMessage.content
                : "Issue detected in pipeline",
              fix: originalMessage.codeSnippet || "Fix applied",
              timestamp: new Date(),
              automatic: false,
              templateId: originalMessage.templateId,
              critical: originalMessage.critical,
            },
            ...prev,
          ])

          // Create persistent notification for critical fixes
          if (originalMessage.critical) {
            toast.warning("Critical Fix Applied", {
              description: "A critical fix has been applied. System monitoring is recommended.",
              persistent: true,
              priority: "high", // Set to high priority
              action: {
                label: "View Details",
                onClick: () => {
                  setActiveTab("fixes")
                  setIsOpen(true)
                },
              },
            })
          }
        }
      }, 500)
    }
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => {
        // Use toast notification instead of console.log
        toast.success("Copied to clipboard", {
          duration: 2000,
        })
      })
      .catch((err) => {
        console.error("Failed to copy: ", err)
        toast.error("Failed to copy to clipboard")
      })
  }

  const applyAutoFix = (error: Error) => {
    // Mark the error as fixed
    setRecentErrors((prev) => prev.map((e) => (e.id === error.id ? { ...e, autoFixed: true, fixApplied: true } : e)))

    // Add system message about the auto-fix
    const autoFixMessage: Message = {
      id: messages.length + 1,
      content: `Auto-fix applied: ${error.message}`,
      sender: "system",
      timestamp: new Date(),
      type: "auto-fix",
      autoFixed: true,
      codeSnippet: generateFixCode(error.message),
    }

    setMessages((prev) => [...prev, autoFixMessage])

    // Add to applied fixes
    const newFix = {
      id: appliedFixes.length + 1,
      error: error.message,
      fix: generateFixDescription(error.message),
      timestamp: new Date(),
      automatic: true,
      critical: error.severity === "high",
    }

    setAppliedFixes((prev) => [newFix, ...prev])

    // Create persistent notification for high severity auto-fixes
    if (error.severity === "high") {
      toast.info("High Severity Auto-Fix Applied", {
        description: `Auto-fix applied: ${error.message}`,
        persistent: true,
        priority: "medium", // Set to medium priority
        action: {
          label: "View Details",
          onClick: () => {
            setActiveTab("fixes")
            setIsOpen(true)
          },
        },
      })
    }
  }

  const handleApplyTemplate = (template: FixTemplate) => {
    // Check if this is a critical template
    const isCritical = template.tags?.includes("critical") || false

    // Add message about applying the template
    const templateMessage: Message = {
      id: messages.length + 1,
      content: `Applying fix template: ${template.name}`,
      sender: "system",
      timestamp: new Date(),
      type: "template",
      codeSnippet: template.code,
      templateId: template.id,
      critical: isCritical,
      actions: [
        { label: "Apply", action: "apply" },
        { label: "Edit", action: "edit" },
      ],
    }

    setMessages((prev) => [...prev, templateMessage])

    // Create persistent notification for critical templates
    if (isCritical) {
      toast.warning(`Critical Template: ${template.name}`, {
        description: "This template addresses a critical issue and requires monitoring.",
        persistent: true,
        action: {
          label: "View Details",
          onClick: () => {
            setActiveTab("chat")
            setIsOpen(true)
          },
        },
      })
    }

    // Scroll to the new message
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, 100)
  }

  const generateFixCode = (errorMessage: string): string => {
    // Generate appropriate fix code based on the error message
    if (errorMessage.includes("schema mismatch") || errorMessage.includes("type mismatch")) {
      return "ALTER TABLE affected_table\nALTER COLUMN affected_column TYPE appropriate_type;"
    } else if (errorMessage.includes("missing values")) {
      return "UPDATE table_name\nSET column_name = default_value\nWHERE column_name IS NULL;"
    } else if (errorMessage.includes("drift")) {
      return "// Updating drift threshold\nconfig.setDriftThreshold('feature_name', new_threshold);"
    } else {
      return "// Applying standard fix procedure\nfixPipeline.applyStandardFix(errorType);"
    }
  }

  const generateFixDescription = (errorMessage: string): string => {
    // Generate a human-readable description of the fix
    if (errorMessage.includes("schema mismatch") || errorMessage.includes("type mismatch")) {
      return "Updated column data type to match expected schema"
    } else if (errorMessage.includes("missing values")) {
      return "Applied default values to null entries"
    } else if (errorMessage.includes("drift")) {
      return "Adjusted drift threshold and recalibrated feature monitoring"
    } else {
      return "Applied standard fix procedure based on error pattern"
    }
  }

  const shouldAutoFix = (error: Error): boolean => {
    // Check if this error should be auto-fixed based on settings
    if (!autoFixSettings.enabled) return false
    if (!error.autoFixable) return false

    // Check severity level settings
    if (!autoFixSettings.severityLevels[error.severity]) return false

    // For simulation purposes, we'll use a random number to represent confidence
    const confidence = Math.floor(Math.random() * 100)
    return confidence >= autoFixSettings.confidenceThreshold
  }

  const processUserInput = (input: string) => {
    setInput(input);
    setActiveTab("chat");
  };

  useEffect(() => {
    if (messagesEndRef.current && !isMinimized) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages, isMinimized])

  // Simulate new errors coming in
  useEffect(() => {
    const interval = setInterval(() => {
      const random = Math.random()
      if (random > 0.7) {
        if (!isOpen) {
          setNotifications((prev) => prev + 1)
        }

        const errorTypes = [
          "Schema validation failed for column",
          "Drift detected in feature",
          "Missing values in required field",
          "Type mismatch in column",
        ]

        const columns = ["property_value", "claim_amount", "location_code", "risk_factor", "policy_number"]

        const newError: Error = {
          id: Date.now(),
          message: `${errorTypes[Math.floor(Math.random() * errorTypes.length)]} '${columns[Math.floor(Math.random() * columns.length)]}'`,
          timestamp: new Date(),
          severity: ["low", "medium", "high"][Math.floor(Math.random() * 3)] as "low" | "medium" | "high",
          autoFixable: Math.random() > 0.3, // 70% of errors are auto-fixable
          critical: Math.random() > 0.8, // 20% chance of being critical
        }

        setRecentErrors((prev) => [newError, ...prev.slice(0, 9)])

        // Check if this error should be auto-fixed
        if (shouldAutoFix(newError)) {
          // If notifyBeforeFix is enabled, add a notification message first
          if (autoFixSettings.notifyBeforeFix) {
            const notificationMessage: Message = {
              id: messages.length + 1,
              content: `Auto-fixable issue detected: ${newError.message}. Applying fix...`,
              sender: "system",
              timestamp: new Date(),
              type: "warning",
            }
            setMessages((prev) => [...prev, notificationMessage])
          }

          // Apply the auto-fix after a short delay
          setTimeout(() => {
            applyAutoFix(newError)
          }, 2000)
        } else if (newError.critical) {
          // Create persistent notification for critical errors that can't be auto-fixed
          toast.error("Critical Error Detected", {
            persistent: true,
            group: "Chat Errors",
            source: "Error & Fix Chat",
            priority: "critical", // Set to critical priority
            description: newError.message,
            action: {
              label: "View Error",
              onClick: () => {
                setActiveTab("errors")
                setIsOpen(true)
              },
            },
          })
        }
      }
    }, 30000) // Every 30 seconds

    return () => clearInterval(interval)
  }, [isOpen, messages.length, autoFixSettings, toast])

  return (
    <>
      {!isOpen ? (
        <Button
          onClick={() => {
            setIsOpen(true)
            setNotifications(0)
          }}
          className="fixed bottom-6 right-6 h-14 w-14 rounded-full shadow-lg bg-primary hover:bg-primary/90 text-primary-foreground"
          aria-label="Open chat assistant"
        >
          <div className="relative">
            <Keyboard className="h-6 w-6" />
            {notifications > 0 && (
              <Badge className="absolute -top-2 -right-2 h-5 w-5 p-0 flex items-center justify-center bg-destructive text-destructive-foreground">
                {notifications}
              </Badge>
            )}
          </div>
        </Button>
      ) : (
        <div
          className={cn(
            "fixed inset-x-0 bottom-0 z-50 transition-all duration-300 ease-in-out md:right-6 md:left-auto md:w-96",
            isMinimized ? "h-14" : "h-[70vh] max-h-[600px]",
          )}
        >
          <Card className="h-full rounded-b-none md:rounded-b-lg border-b-0 md:border-b shadow-lg">
            <CardHeader className="py-2 px-4 cursor-move bg-muted/50 border-b flex flex-row items-center justify-between">
              <CardTitle className="text-sm font-medium">Error & Fix Assistant</CardTitle>
              <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setIsOpen(false)}>
                <X className="h-4 w-4" />
              </Button>
            </CardHeader>

            {!isMinimized && (
              <>
                <Tabs
                  defaultValue="chat"
                  value={activeTab}
                  onValueChange={setActiveTab}
                  className="flex flex-col h-[calc(100%-7rem)]"
                >
                  <TabsList className="mx-3 mt-2 mb-0">
                    <TabsTrigger value="chat" className="flex-1">
                      Chat
                    </TabsTrigger>
                    <TabsTrigger value="errors" className="flex-1">
                      Recent Errors
                    </TabsTrigger>
                    <TabsTrigger value="fixes" className="flex-1">
                      Applied Fixes
                    </TabsTrigger>
                    <TabsTrigger value="auto-fix" className="flex-1">
                      Auto-Fix
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="chat" className="flex-1 overflow-hidden p-0 m-0">
                    <CardContent className="p-0 overflow-hidden h-full">
                      <ScrollArea className="h-full p-4">
                        <div className="space-y-4">
                          {messages.map((message) => (
                            <div
                              key={message.id}
                              className={cn(
                                "flex flex-col max-w-[85%] rounded-lg p-3 text-sm",
                                message.sender === "user"
                                  ? "ml-auto bg-primary text-primary-foreground"
                                  : message.sender === "system"
                                    ? "bg-muted/80 border border-muted-foreground/20"
                                    : "bg-muted",
                                message.type === "error" &&
                                  message.sender !== "user" &&
                                  "border-l-4 border-destructive",
                                message.type === "success" &&
                                  message.sender !== "user" &&
                                  "border-l-4 border-green-500",
                                message.type === "fix" && message.sender !== "user" && "border-l-4 border-yellow-500",
                                message.type === "warning" &&
                                  message.sender !== "user" &&
                                  "border-l-4 border-orange-500",
                                message.type === "auto-fix" && "border-l-4 border-blue-500",
                                message.type === "template" && "border-l-4 border-purple-500",
                                message.critical && "bg-red-50 dark:bg-red-900/20",
                              )}
                            >
                              {message.type === "auto-fix" && (
                                <div className="flex items-center mb-1 text-xs text-blue-500 font-medium">
                                  <Zap className="h-3 w-3 mr-1" />
                                  AUTO-FIX APPLIED
                                </div>
                              )}

                              {message.type === "template" && (
                                <div className="flex items-center mb-1 text-xs text-purple-500 font-medium">
                                  <FileCode className="h-3 w-3 mr-1" />
                                  FIX TEMPLATE
                                </div>
                              )}

                              {message.critical && (
                                <div className="flex items-center mb-1 text-xs text-red-500 font-medium">
                                  <AlertTriangle className="h-3 w-3 mr-1" />
                                  CRITICAL ISSUE
                                </div>
                              )}

                              <div className="whitespace-pre-wrap">{message.content}</div>

                              {message.codeSnippet && (
                                <div className="mt-2 relative">
                                  <pre className="bg-background text-foreground p-2 rounded text-xs overflow-x-auto">
                                    <code>{message.codeSnippet}</code>
                                  </pre>
                                  <Button
                                    variant="ghost"
                                    size="icon"
                                    className="absolute top-1 right-1 h-6 w-6 opacity-70 hover:opacity-100"
                                    onClick={() => copyToClipboard(message.codeSnippet || "")}
                                  >
                                    <Copy className="h-3 w-3" />
                                  </Button>
                                </div>
                              )}

                              {message.actions && message.actions.length > 0 && (
                                <div className="flex flex-wrap gap-2 mt-2">
                                  {message.actions.map((action, index) => (
                                    <Button
                                      key={index}
                                      size="sm"
                                      variant={
                                        action.action === "approve" || action.action === "apply" ? "default" : "outline"
                                      }
                                      className={cn("text-xs", action.applied && "opacity-50 pointer-events-none")}
                                      onClick={() => handleActionClick(message.id, index)}
                                      disabled={action.applied}
                                    >
                                      {action.applied ? (
                                        <>
                                          <Check className="h-3 w-3 mr-1" />
                                          {action.label}
                                        </>
                                      ) : (
                                        action.label
                                      )}
                                    </Button>
                                  ))}
                                </div>
                              )}

                              <span className="text-xs opacity-70 mt-1">
                                {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                            </div>
                          ))}
                          <div ref={messagesEndRef} />
                        </div>
                      </ScrollArea>
                    </CardContent>
                  </TabsContent>

                  <TabsContent value="errors" className="flex-1 overflow-hidden p-0 m-0">
                    <CardContent className="p-0 overflow-hidden h-full">
                      <ScrollArea className="h-full p-4">
                        <div className="space-y-3">
                          {recentErrors.map((error) => (
                            <div
                              key={error.id}
                              className={cn(
                                "bg-muted p-3 rounded-lg text-sm",
                                error.fixApplied && "opacity-60",
                                error.critical &&
                                  !error.fixApplied &&
                                  "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800",
                              )}
                            >
                              <div className="flex items-start justify-between">
                                <div className="flex-1">
                                  <div className="font-medium flex items-center flex-wrap gap-2">
                                    <Badge
                                      variant="outline"
                                      className={cn(
                                        error.severity === "high" && "bg-destructive text-destructive-foreground",
                                        error.severity === "medium" && "bg-yellow-500 text-yellow-950",
                                        error.severity === "low" && "bg-blue-500 text-blue-950",
                                      )}
                                    >
                                      {error.severity}
                                    </Badge>

                                    {error.critical && (
                                      <Badge variant="outline" className="bg-red-500 text-red-950">
                                        critical
                                      </Badge>
                                    )}

                                    {error.message}

                                    {error.autoFixable && !error.fixApplied && (
                                      <Badge
                                        variant="outline"
                                        className="bg-blue-500/10 text-blue-500 border-blue-500/30"
                                      >
                                        auto-fixable
                                      </Badge>
                                    )}

                                    {error.autoFixed && (
                                      <Badge className="bg-green-500 text-green-950">
                                        <Check className="h-3 w-3 mr-1" />
                                        auto-fixed
                                      </Badge>
                                    )}

                                    {error.fixApplied && !error.autoFixed && (
                                      <Badge className="bg-green-500 text-green-950">
                                        <Check className="h-3 w-3 mr-1" />
                                        fixed
                                      </Badge>
                                    )}
                                  </div>
                                  <div className="text-xs opacity-70 mt-1">{error.timestamp.toLocaleString()}</div>
                                </div>

                                {!error.fixApplied && (
                                  <div className="flex gap-1">
                                    {error.autoFixable && (
                                      <Button
                                        variant="outline"
                                        size="icon"
                                        className="h-6 w-6 bg-blue-500/10 text-blue-500 border-blue-500/30 hover:bg-blue-500/20 hover:text-blue-600"
                                        onClick={() => applyAutoFix(error)}
                                      >
                                        <Zap className="h-3 w-3" />
                                      </Button>
                                    )}
                                    <Button
                                      variant="ghost"
                                      size="icon"
                                      className="h-6 w-6"
                                      onClick={() => {
                                        setActiveTab("chat")
                                        processUserInput(`Fix error: ${error.message}`)
                                      }}
                                    >
                                      <Terminal className="h-3 w-3" />
                                    </Button>
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}

                          {recentErrors.length === 0 && (
                            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                              <Check className="h-8 w-8 mb-2" />
                              <p>No recent errors detected</p>
                            </div>
                          )}
                        </div>
                      </ScrollArea>
                    </CardContent>
                  </TabsContent>

                  <TabsContent value="fixes" className="flex-1 overflow-hidden p-0 m-0">
                    <CardContent className="p-0 overflow-hidden h-full">
                      <ScrollArea className="h-full p-4">
                        <div className="space-y-3">
                          {appliedFixes.map((fix) => (
                            <div
                              key={fix.id}
                              className={cn(
                                "bg-muted p-3 rounded-lg text-sm",
                                fix.critical &&
                                  "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800",
                              )}
                            >
                              <div className="flex items-start justify-between">
                                <div>
                                  <div className="font-medium flex items-center flex-wrap gap-2">
                                    {fix.automatic ? (
                                      <Badge className="mr-2 bg-blue-500 text-blue-950">
                                        <Zap className="h-3 w-3 mr-1" />
                                        Auto
                                      </Badge>
                                    ) : (
                                      <Badge className="mr-2 bg-green-500 text-green-950">
                                        <Check className="h-3 w-3 mr-1" />
                                        Manual
                                      </Badge>
                                    )}
                                    {fix.templateId && (
                                      <Badge className="mr-2 bg-purple-500 text-purple-950">
                                        <FileCode className="h-3 w-3 mr-1" />
                                        Template
                                      </Badge>
                                    )}
                                    {fix.critical && (
                                      <Badge className="mr-2 bg-red-500 text-red-950">
                                        <AlertTriangle className="h-3 w-3 mr-1" />
                                        Critical
                                      </Badge>
                                    )}
                                    {fix.error}
                                  </div>
                                  <div className="text-xs mt-1">{fix.fix}</div>
                                  <div className="text-xs opacity-70 mt-1 flex items-center">
                                    <Clock className="h-3 w-3 mr-1" />
                                    {fix.timestamp.toLocaleString()}
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}

                          {appliedFixes.length === 0 && (
                            <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                              <AlertTriangle className="h-8 w-8 mb-2" />
                              <p>No fixes have been applied yet</p>
                            </div>
                          )}
                        </div>
                      </ScrollArea>
                    </CardContent>
                  </TabsContent>

                  <TabsContent value="auto-fix" className="flex-1 overflow-hidden p-0 m-0">
                    <CardContent className="p-4 overflow-hidden h-full">
                      <ScrollArea className="h-full pr-4">
                        <div className="space-y-4">
                          <div>
                            <div className="flex items-center justify-between mb-2">
                              <h3 className="text-sm font-medium">Auto-Fix System</h3>
                              <Switch
                                checked={autoFixSettings.enabled}
                                onCheckedChange={(checked) =>
                                  setAutoFixSettings((prev) => ({ ...prev, enabled: checked }))
                                }
                              />
                            </div>
                            <p className="text-xs text-muted-foreground mb-4">
                              When enabled, the system will automatically fix certain errors without requiring manual
                              approval.
                            </p>
                          </div>

                          <div className="space-y-3">
                            <h3 className="text-sm font-medium">Confidence Threshold</h3>
                            <p className="text-xs text-muted-foreground">
                              Only apply fixes when confidence is above:{" "}
                              <span className="font-medium">{autoFixSettings.confidenceThreshold}%</span>
                            </p>
                            <Slider
                              value={[autoFixSettings.confidenceThreshold]}
                              min={50}
                              max={100}
                              step={5}
                              onValueChange={(value) =>
                                setAutoFixSettings((prev) => ({ ...prev, confidenceThreshold: value[0] }))
                              }
                              disabled={!autoFixSettings.enabled}
                            />
                          </div>

                          <div className="space-y-3">
                            <h3 className="text-sm font-medium">Error Severity Levels</h3>
                            <p className="text-xs text-muted-foreground">Select which severity levels to auto-fix:</p>

                            <div className="space-y-2">
                              <div className="flex items-center justify-between">
                                <div className="flex items-center">
                                  <Badge variant="outline" className="bg-blue-500 text-blue-950 mr-2">
                                    low
                                  </Badge>
                                  <span className="text-xs">Low severity errors</span>
                                </div>
                                <Switch
                                  checked={autoFixSettings.severityLevels.low}
                                  onCheckedChange={(checked) =>
                                    setAutoFixSettings((prev) => ({
                                      ...prev,
                                      severityLevels: { ...prev.severityLevels, low: checked },
                                    }))
                                  }
                                  disabled={!autoFixSettings.enabled}
                                />
                              </div>

                              <div className="flex items-center justify-between">
                                <div className="flex items-center">
                                  <Badge variant="outline" className="bg-yellow-500 text-yellow-950 mr-2">
                                    medium
                                  </Badge>
                                  <span className="text-xs">Medium severity errors</span>
                                </div>
                                <Switch
                                  checked={autoFixSettings.severityLevels.medium}
                                  onCheckedChange={(checked) =>
                                    setAutoFixSettings((prev) => ({
                                      ...prev,
                                      severityLevels: { ...prev.severityLevels, medium: checked },
                                    }))
                                  }
                                  disabled={!autoFixSettings.enabled}
                                />
                              </div>

                              <div className="flex items-center justify-between pt-2">
                            <div>
                              <h3 className="text-sm font-medium">Notify Before Fixing</h3>
                              <p className="text-xs text-muted-foreground">
                                Send a notification before applying auto-fixes
                              </p>
                            </div>
                            <Switch
                              checked={autoFixSettings.notifyBeforeFix}
                              onCheckedChange={(checked) =>
                                setAutoFixSettings((prev) => ({ ...prev, notifyBeforeFix: checked }))
                              }
                              disabled={!autoFixSettings.enabled}
                            />
                          </div>
                        </div>
                      </ScrollArea>
                    </CardContent>
                  </TabsContent>
                </Tabs>

                <CardFooter className="p-3 pt-0 border-t">
                  <div className="flex w-full items-center space-x-2">
                    <Input
                      placeholder="Ask about errors or fixes..."
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleSend()}
                      className="flex-1"
                    />
                    <FixTemplatesDialog onApplyTemplate={handleApplyTemplate} />
                    <Button size="icon" onClick={handleSend}>
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </CardFooter>
              </>
            )}
          </Card>
        </div>
      )}
    </>
  );
}
