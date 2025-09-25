import time
import json
import logging
from collections import deque
import re
import uuid
import random
import itertools
import matplotlib.pyplot as plt

# --- Enhanced EventLogger ---
class EventLogger:
    def __init__(self):
        self.events = deque(maxlen=1000)
        logging.basicConfig(
            filename='symbolic_runtime.log',
            level=logging.INFO,
            format='%(asctime)s - %(component)s - %(levelname)s - %(message)s'
        )

    def log(self, timestamp, component, event_type, status, message):
        event_id = str(uuid.uuid4())
        event = {
            'event_id': event_id,
            'timestamp': timestamp,
            'component': component,
            'event_type': event_type,
            'status': status,
            'message': message
        }
        self.events.append(event)
        logging.info(message, extra={'component': component})

    def get_events(self):
        return list(self.events)

    def print_symbolic_logs(self):
        print("\n--- Symbolic Event Log ---")
        for event in self.events:
            print(f"  [{event['timestamp']:.2f}] {event['component']:<12} {event['event_type']:<15} {event['status']:<10} {event['message']}")
        print("--------------------------")

# --- Core Agent Class ---
class SymbolicRuntime:
    def __init__(self, keys=None, unlock_order=None, log_file=None, intermittent_failure_rate=0.0, automatic_recovery=True, resource_limits=None):
        self.keys = keys or {}
        self.unlock_order = unlock_order or ['Hardware', 'Kernel', 'Applications']
        self.applications = []
        self.event_logger = EventLogger()
        self.symbolic_system_structure = {}
        self.log_file = log_file
        self.unlock_plan = None
        self.success_criteria = {
            'Hardware': lambda status: status == 'Available',
            'Kernel': lambda status, modules_loaded: status == 'Available' and modules_loaded,
            'Applications': lambda status, apps_running: status == 'Available' and apps_running,
        }
        self.successful_steps = 0
        self.failed_steps = 0
        self.intermittent_failure_rate = intermittent_failure_rate
        self.faults_injected = 0
        self.recovery_performed = 0
        self.automatic_recovery = automatic_recovery
        self.current_state = {}
        self.resource_limits = resource_limits or {}
        self.dynamic_dependencies = {}
        self.test_results = [] #For Visualization

    def _log_event(self, timestamp, component, event_type, status, message):
        self.event_logger.log(timestamp, component, event_type, status, message)

    def passive_scan(self, symbolic_system):
        """Performs passive scan, identifies structure and dependencies, and sets resource limits"""
        print("\n🔍 Performing Passive Scan...")
        timestamp = time.time()
        self.symbolic_system_structure = {}
        self.current_state = {}
        self.dynamic_dependencies = {}  # Reset dynamic dependencies

        self.symbolic_system_structure['Hardware'] = "Present"
        self.current_state['Hardware'] = "Present"  # Set initial state
        self._log_event(timestamp, 'Hardware', 'Discovery', 'Present', 'Hardware layer detected.')

        # Check for Kernel modules
        try:
            if hasattr(symbolic_system, 'loaded_modules') and isinstance(symbolic_system.loaded_modules, list):
                self.symbolic_system_structure['Kernel'] = "Modules Loaded"
                modules_str = ", ".join(symbolic_system.loaded_modules)
                self.current_state['Kernel'] = "Modules Loaded"
                # Simulate dynamic dependency discovery and resource limits for modules
                for module in symbolic_system.loaded_modules:
                    self.dynamic_dependencies[module] = 'Hardware'  # Example dependency
                    if module not in self.resource_limits:
                        self.resource_limits[module] = random.randint(30, 80)  # Example resource limit
                        self._log_event(timestamp, 'Runtime', 'ResourceLimit', 'Discovered', f"Discovered resource limit for {module}: {self.resource_limits[module]}")

                self._log_event(timestamp, 'Kernel', 'Discovery', 'Modules Found', f"Kernel modules: {modules_str}")
            else:
                self.symbolic_system_structure['Kernel'] = "No Modules"
                self.current_state['Kernel'] = "No Modules"
                self._log_event(timestamp, 'Kernel', 'Discovery', 'No Modules', "No kernel modules detected.")
        except Exception as e:
            self.symbolic_system_structure['Kernel'] = "Error"
            self.current_state['Kernel'] = "Error"
            self._log_event(timestamp, 'Kernel', 'Discovery', 'Error', f"Error detecting Kernel modules: {e}")

        # Check for Applications
        if self.applications:
            self.symbolic_system_structure['Applications'] = "Applications Found"
            app_names = [app.name for app in self.applications]
            apps_str = ", ".join(app_names)
            self.current_state['Applications'] = "Applications Found"
            # Simulate dynamic dependency discovery and resource limits for applications
            for app in self.applications:
                 self.dynamic_dependencies[app.name] = 'Kernel'  # Example dependency
                 if app.name not in self.resource_limits:
                    self.resource_limits[app.name] = random.randint(50, 100)  # Example resource limit
                    self._log_event(timestamp, 'Runtime', 'ResourceLimit', 'Discovered', f"Discovered resource limit for {app.name}: {self.resource_limits[app.name]}")

            self._log_event(timestamp, 'Applications', 'Discovery', 'Applications Found', f"Applications detected: {apps_str}")
        else:
            self.symbolic_system_structure['Applications'] = "No Applications"
            self.current_state['Applications'] = "No Applications"
            self._log_event(timestamp, 'Applications', 'Discovery', 'No Applications', "No applications detected.")

        self._log_event(timestamp, 'Runtime', 'Scan', 'Complete', 'Passive scan completed.')
        print("✅ Passive scan completed.")

    def analyze_access_points(self, symbolic_system):
        """Identifies access points, keys, and dependencies dynamically."""
        print("\n🔑 Analyzing Access Points...")
        timestamp = time.time()

        # Check each layer in order
        for layer in self.unlock_order:
            required_key = self.keys.get(layer, None)

            if required_key:
                msg = f"Layer {layer} requires key: {required_key}"
                status = "Key Available" if required_key else "Missing Key"
                self._log_event(timestamp, layer, 'AccessPoint', status, msg)
                print(f"🔐 {msg}")
            else:
                msg = f"Layer {layer} does not require a key"
                self._log_event(timestamp, layer, 'AccessPoint', 'Open', msg)
                print(f"✅ {msg}")

            # Drill down into subcomponents (e.g., modules or applications)
            if layer == 'Kernel':
                modules = getattr(symbolic_system, 'loaded_modules', [])
                for module in modules:
                    module_msg = f"Kernel module {module} is accessible once Kernel is unlocked"
                    self._log_event(timestamp, layer, 'AccessPoint', 'Dependent', module_msg)
                    print(f"📦 {module_msg}")

            elif layer == 'Applications':
                apps = getattr(self, 'applications', [])
                for app in apps:
                    app_msg = f"Application {app.name} is accessible once Applications are unlocked"
                    self._log_event(timestamp, layer, 'AccessPoint', 'Dependent', app_msg)
                    print(f"🚀 {app_msg}")

        self._log_event(timestamp, 'Runtime', 'AccessPointAnalysis', 'Complete', 'Access point analysis completed.')

    def plan_unlock_order(self, symbolic_system):
        """Formulates a sequential plan to unlock the system, incorporating dependencies."""
        print("\n📝 Planning Unlock Order...")
        timestamp = time.time()
        unlock_plan = []

        if not self.symbolic_system_structure:
            self.passive_scan(symbolic_system)

        # Add layers to the unlock plan
        for layer in self.unlock_order:
            required_key = self.keys.get(layer, None)
            status = "Key Present" if required_key else "Missing Key"
            action = "Attempt Unlock"
            dependencies = []

            if layer == 'Kernel' and 'Hardware' in self.symbolic_system_structure:
                dependencies.append('Hardware')
            elif layer == 'Applications' and 'Kernel' in self.symbolic_system_structure:
                dependencies.append('Kernel')

            plan_entry = {
                'action': action,
                'layer': layer,
                'status': status,
                'key_status': required_key if required_key else "Missing",
                'dependencies': dependencies
            }
            unlock_plan.append(plan_entry)

        # Add subcomponent actions (load modules, launch apps), incorporating dependencies
        if 'Kernel' in self.symbolic_system_structure and self.symbolic_system_structure['Kernel'] == "Modules Loaded":
            for module in symbolic_system.loaded_modules:
                if module in self.dynamic_dependencies:
                    unlock_plan.append({
                         'action': 'Load module',
                         'layer': 'Kernel',
                         'module': module,
                         'status': 'Pending',
                         'key_status': 'N/A',
                         'dependencies': [self.dynamic_dependencies[module]]
                    })

        if 'Applications' in self.symbolic_system_structure and self.symbolic_system_structure['Applications'] == "Applications Found":
            for app in self.applications:
                if app.name in self.dynamic_dependencies:
                    unlock_plan.append({
                         'action': 'Launch',
                         'layer': 'Applications',
                         'app': app.name,
                         'status': 'Pending',
                         'key_status': 'N/A',
                         'dependencies': [self.dynamic_dependencies[app.name]]
                    })

        self._log_event(timestamp, 'Runtime', 'Plan', 'Complete', 'Unlock plan formulated.')
        self.unlock_plan = unlock_plan
        print("✅ Unlock plan formulated successfully.")
        print("\n--- Detailed Unlock Plan ---")
        for i, entry in enumerate(unlock_plan):
            print(f"  Step {i + 1}:")
            for key, value in entry.items():
                print(f"    {key}: {value}")
            print("---")

        return unlock_plan

    def symbolic_query(self, command: str):
        """Executes a symbolic query, now with dynamic dependencies."""
        timestamp = time.time()
        action, *target_parts = command.split(" ", 1)
        target = target_parts[0] if target_parts else None
        success = False
        status = "Unknown"

        result = None

        if action.lower() == "unlock" and target:
            if random.random() < self.intermittent_failure_rate:
                result = f"❌ Failed to unlock {target} due to intermittent issue."
                self._log_event(timestamp, target, 'Execute', 'Failure', result)
                status = "Unavailable"
                self.current_state[target] = "Unavailable"
                print(f"💻 {result}")
                return False, result, status

            result = f" {target} unlocked symbolically"
            self._log_event(timestamp, target, 'Execute', 'Unlocked', result)
            status = "Available"
            self.current_state[target] = "Available"
            success = True

        elif action.lower() == "status" and target:
            if random.random() < self.intermittent_failure_rate:
                result = f"❌ Could not retrieve status of {target} due to intermittent issue."
                self._log_event(timestamp, target, 'Query', 'Intermittent Failure', result)
                status = "Unknown"
                print(f"💻 {result}")
                return False, result, status

            result = f"Status of {target}: Available"
            self._log_event(timestamp, target, 'Query', 'Status', result)
            status = "Available"
            success = True

        elif action.lower() == "run" and target:
            # Check for resource limits
            if target in self.resource_limits:
                resource_usage = random.randint(1, 100)  # Simulate resource usage
                if resource_usage > self.resource_limits[target]:
                    result = f"❌ Application {target} failed to run due to resource exhaustion (required: {resource_usage}, limit: {self.resource_limits[target]})"
                    self._log_event(timestamp, target, 'Execute', 'ResourceExhaustion', result)
                    status = "Failed"
                    print(f"💻 {result}")
                    return False, result, status

            result = f"Application {target} executed"
            self._log_event(timestamp, target, 'Execute', 'Run', result)
            status = "Running"
            success = True

        else:
            result = f"Unknown command: {command}"
            self._log_event(timestamp, 'System', 'Error', 'Invalid', result)
            status = "Invalid"

        print(f"💻 {result}")
        return success, result, status

    def execute_plan(self, plan):
        """Executes the unlock plan, handles retries and recovery, and considers dynamic dependencies."""
        print("\n▶️ Executing Unlock Plan...")
        results = []
        successful_steps = 0
        failed_steps = 0
        modules_loaded = False
        apps_running = False
        recovered_steps = 0

        for step in plan:
            layer = step.get('layer')
            action = step.get('action')
            module = step.get('module')
            app = step.get('app')
            original_key = self.keys.get(layer)
            max_retries = 2
            retry_delay = 1

            for attempt in range(max_retries + 1):
                # -- Attempt Unlock
                if action == 'Attempt Unlock':
                    success, result, status = self.symbolic_query(f"unlock {layer}")
                    results.append(result)

                    if success:
                        if layer in self.success_criteria:
                            if layer == 'Kernel':
                                if self.symbolic_system_structure.get('Kernel') == "Modules Loaded":
                                    modules_loaded = True
                                    success = self.success_criteria[layer](status, modules_loaded)
                                else:
                                    success = False
                            elif layer == 'Applications':
                                if self.symbolic_system_structure.get('Applications') == "Applications Found":
                                    apps_running = True
                                    success = self.success_criteria[layer](status, apps_running)
                                else:
                                    success = False
                            else:
                                success = self.success_criteria[layer](status)

                        if success:
                            successful_steps += 1
                            self._log_event(time.time(), layer, 'Execute', 'Success', f"Step '{layer}' met success criteria after {attempt + 1} attempts.")
                            step['status'] = "Success"
                            break
                        else:
                            failed_steps += 1
                            self._log_event(time.time(), layer, 'Execute', 'Failure', f"Step '{layer}' did not meet success criteria after {attempt + 1} attempts.")
                            step['status'] = "Failure"
                            if self.automatic_recovery and original_key and self.keys.get(layer) != original_key:
                                # Simulate key recovery
                                self.keys[layer] = original_key
                                self._log_event(time.time(), layer, 'Recovery', 'KeyRestored', f"Key for {layer} restored to its original value. Retrying...")
                                recovered_steps += 1

                            if attempt < max_retries:
                                print(f"   Retrying step {layer} in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            else:
                                print(f"   Max retries reached for step {layer}. Moving on.")
                                break

                    else:
                        failed_steps += 1
                        self._log_event(time.time(), layer, 'Execute', 'Failure', f"Step '{layer}' failed. Attempt {attempt + 1} of {max_retries + 1}.")
                        step['status'] = "Failure"
                        if self.automatic_recovery:
                            # Attempt recovery
                            if original_key and self.keys.get(layer) != original_key:
                                # Simulate key recovery
                                self.keys[layer] = original_key
                                self._log_event(time.time(), layer, 'Recovery', 'KeyRestored', f"Key for {layer} restored to its original value. Retrying...")
                                recovered_steps += 1

                        if attempt < max_retries:
                            print(f"   Retrying step {layer} in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print(f"   Max retries reached for step {layer}. Moving on.")
                            break

                # Load Module or Launch App
                elif action in ['Load module', 'Launch']:
                    target = module if action == "Load module" else app
                    success, result, status = self.symbolic_query(f"{action} {target}")
                    results.append(result)
                    dep_met = True
                    dep_layer = layer if action == "Load module" else "Applications"
                    #Check and verify dynamic dependencies as a check
                    if(self.dynamic_dependencies[target] not in self.current_state and self.current_state.get(dep_layer) != "Available"):
                      dep_met = False

                    if success and dep_met:
                        successful_steps += 1
                        step['status'] = "Success"
                        if action == "Load module":
                          modules_loaded = True #set global to true
                        elif action == "Launch":
                          apps_running = True #set global to true
                        break

                    else:
                        failed_steps += 1
                        step['status'] = "Failure"
                        #More granular error handling can be done
                        #We can create dependency fixes as well.
                        if self.automatic_recovery and action == 'Launch' and status != "Available" :
                            if dep_layer == 'Kernel':
                                  self._log_event(time.time(), layer, 'Execute', 'Retrying', f"Reloading Kernel {layer}")
                                  modules_loaded = True #set back
                                  #reinject that if the user had set it before with previous runs
                                  success, result, status = self.symbolic_query(f"{action} {target}") #try running the module after dependencies
                                  break

                        if attempt < max_retries:
                            print(f"   Retrying step {target} in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            print(f"   Max retries reached for step {target}. Moving on.")
                            break
                else:
                    results.append("Skipped")  # Handle unexpected steps
                    step['status'] = "Skipped"
                break

        self.successful_steps = successful_steps
        self.failed_steps = failed_steps
        self.recovery_performed = recovered_steps
        return results

    def generate_report(self):
        """Generates a report summarizing the simulation results."""
        timestamp = time.time()
        total_steps = len(self.unlock_plan)
        successful_steps = sum(1 for step in self.unlock_plan if step['status'] == "Success")
        failed_steps = sum(1 for step in self.unlock_plan if step['status'] == "Failure")

        success_rate = (successful_steps / total_steps) * 100 if total_steps else 0

        report = {
            'Total Steps': total_steps,
            'Successful Steps': successful_steps,
            'Failed Steps': failed_steps,
            'Recovery Performed': self.recovery_performed,
            'Faults Injected': self.faults_injected,
            'Intermittent Failure Rate': self.intermittent_failure_rate,
            'Automatic Recovery': self.automatic_recovery,
            'Current System State': self.current_state,
            'Dynamic Dependencies': self.dynamic_dependencies,
            'Resource Limits': self.resource_limits,
            'Success Rate': f"{success_rate:.2f}%",
            'Errors': [event['message'] for event in self.event_logger.get_events() if event['status'] == 'Error' or "Failure" in event['message']],
            'Unlock Plan': self.unlock_plan
        }

        report_str = json.dumps(report, indent=4)

        self._log_event(timestamp, 'Runtime', 'Report', 'Generated', 'Simulation report generated.')

        print("\n--- Simulation Report ---")
        print(report_str)

        return report

    def print_symbolic_logs(self):
        """Prints the symbolic logs in a formatted way."""
        self.event_logger.print_symbolic_logs()

    def inject_fault(self, target, fault_type, limit=None):
        """Simulates injecting a fault into a specific target."""
        timestamp = time.time()

        if fault_type == "KeyCorrupt":
            if target in self.keys:
                original_key = self.keys[target]
                self.keys[target] = "corrupted_key"  # Simulate key corruption
                self._log_event(timestamp, target, "FaultInjection", "KeyCorrupt", f"Key for {target} corrupted. Original key: {original_key}")
                self.faults_injected += 1

            else:
                self._log_event(timestamp, target, "FaultInjection", "InvalidTarget", f"Cannot corrupt key for {target}: Key not found.")
                return False

        elif fault_type == "ModuleFail":
            if target in self.applications:
                self.applications.remove(target)
                self._log_event(timestamp, target, "FaultInjection", "ModuleFail", f"Module {target} failed to load.")
                self.faults_injected += 1
            elif target in  self.keys :
                original = self.keys.get(target)
                self.keys[target] = "broken"
                self._log_event(timestamp, target, "FaultInjection", "ModuleFail", f"Module {target} set as Fail.")
                self.faults_injected += 1
            else :
                self._log_event(timestamp, target, "FaultInjection", "InvalidTarget", f"Invalid module for {target} to load.")
                return False

        elif fault_type == "Intermittent":
            self.intermittent_failure_rate = 0.5  # set the intermittent failure
            self._log_event(timestamp, "ALL", "FaultInjection", "Intermittent", f"Intermittent failures injected at {self.intermittent_failure_rate} rate.")
            self.faults_injected += 1
        elif fault_type == "ResourceExhaustion":
            if target in self.resource_limits:
               if limit:
                    self.resource_limits[target] = int(limit) #set the resource Limit
                    self._log_event(timestamp, "ALL", "FaultInjection", "ResourceExhaustion", f"Resource limit for {target} at  {limit}.")
               else :
                    print ("must have a limit")
                    return False
            else :
                 return False
            self.faults_injected += 1 #only counts with all correct params

        return True

    def run_interactive_console(self, symbolic_system):
        """Runs an interactive console for manual control and exploration."""
        print("\n--- Interactive Console ---")
        print("Available commands: scan, access, plan, execute, fault, recover, report, logs, exit, generate_test_cases, run_workload, visualize")

        while True:
            command = input("Enter command: ").strip()

            if command == "scan":
                self.passive_scan(symbolic_system)
            elif command == "access":
                self.analyze_access_points(symbolic_system)
            elif command == "plan":
                self.plan_unlock_order(symbolic_system)
            elif command == "execute":
                if self.unlock_plan:
                    self.execute_plan(self.unlock_plan)
                else:
                    print("Error: No unlock plan available. Run 'plan' first.")
            elif command.startswith("fault"):
                try:
                    parts = command.split(" ")
                    if len(parts) >= 3:
                        _, target, fault_type, *extra_params = parts
                        limit = None #set as empty, assume it may happen.
                        if fault_type == "Intermittent":
                            self.inject_fault(target, fault_type)
                            print("Intermittent fault injection active.")

                        elif fault_type == "ResourceExhaustion":
                             if len(parts) > 3 :
                                 limit = parts[3]
                                 self.inject_fault(target, fault_type, limit)

                             else:
                                 print("Invalid fault command. For ResourceExhaustion  command requires a limit. Ex: fault FM ResourceExhaustion 75")
                                 continue

                             print(f"ResourceExhaustion fault for {target} added")
                        else:
                            self.inject_fault(target, fault_type)
                            print("Fault injected.")
                    else:
                        print("Invalid fault command. Usage: fault <target> <fault_type>")
                except ValueError as e:
                    print(f"Error parsing fault command: {e}")
                    print("Fault Injection: Usage: fault <target> <fault_type> or fault <target> <ResourceExhaustion> <limit>")

            elif command == "report":
                self.generate_report()
            elif command == "logs":
                self.print_symbolic_logs()
            elif command == "exit":
                print("Exiting console.")
                break
            elif command == "recover":
                 # Simple automated recovery - re-run the plan
                if self.unlock_plan:
                    print("Attempting automated recovery by re-executing the plan...")
                    self.recovery_performed = 1 #track how many times it was called
                    self.execute_plan(self.unlock_plan) # execute_plan() handles retries
                else:
                    print("Error: No unlock plan available. Run 'plan' first.")

            elif command == "reset":
                #Resets the simulation
                self.recovery_performed = 0
                self.faults_injected = 0
                self.intermittent_failure_rate = 0.0
                self.keys = {
                    'Hardware': 'hwkey',
                    'Kernel': 'kkey',
                    'Applications': 'appkey'
                }
                print("Simulation reset to initial state.")
            elif command.startswith("generate_test_cases"):
                try:
                    parts = command.split(" ")
                    selected_fault_types = []
                    selected_targets = []
                    if len(parts) > 1:
                        fault_params = parts[1].split(",") #split by comma
                        for fault_param in fault_params:
                            if ":" in fault_param:
                                fault_type, target_limit = fault_param.split(":", 1) #split only once to preserve limit value
                                if fault_type == "ResourceExhaustion":
                                    target, limit = target_limit.split(":")
                                    #self.resource_limits[target] = limit  #add a resource limit for test cases.
                                    self.inject_fault(target, fault_type, limit) #Inject the Resource Command during generation
                                selected_fault_types.append(fault_type)
                            else:
                                selected_fault_types.append(fault_param)

                        if len(parts) > 2 :
                            selected_targets = parts[2].split(",")
                    self.generate_test_cases(symbolic_system, selected_fault_types, selected_targets)
                except Exception as e:
                      print(f"generate_test_cases command error {e}")
            elif command.startswith("run_workload"):
                try:
                        _, workload_type = command.split(" ")
                        self.run_workload(workload_type)
                except ValueError:
                        print("Invalid workload command. Usage: run_workload <workload_type>")

            elif command == "visualize":
                self.generate_visualization() #visualize
            else:
                print("Unknown command. See available commands: scan, access, plan, execute, fault, recover, report, logs, exit, generate_test_cases, run_workload")

    def run_workload(self, workload_type):
      """
      Simulate different user workloads.
      """
      print(f"\n--- Running Workload: {workload_type} ---")
      timestamp = time.time()
      if workload_type == "HighLoad":
          #Simulate high CPU and memory usage, and try to run apps
          success, result, status = self.symbolic_query(f"run FileManager")
          self._log_event(timestamp, 'Workload', 'Execute', status, result)
          success, result, status = self.symbolic_query(f"run NetworkMonitor")
          self._log_event(timestamp, 'Workload', 'Execute', status, result)

      elif workload_type == "Normal":
          #Simulate normal usage with some tasks
          success, result, status = self.symbolic_query(f"status Kernel")
          self._log_event(timestamp, 'Workload', 'Query', status, result)

      self._log_event(timestamp, 'Workload', 'Complete', 'Status', f"Workload '{workload_type}' completed.")

    def generate_test_cases(self, symbolic_system, selected_fault_types=None, selected_targets=None):
      """Generates and runs automated test cases with specified fault scenarios."""
      print("\n--- Generating and Running Automated Test Cases ---")
      fault_types = ["KeyCorrupt", "ModuleFail", "Intermittent", "ResourceExhaustion"]
      targets = list(self.keys.keys()) + [app.name for app in self.applications]
      if (selected_targets):
         targets = selected_targets
      if (selected_fault_types):
         fault_types = selected_fault_types #["KeyCorrupt", "ModuleFail", "Intermittent", "ResourceExhaustion"]

      #Generate a test if target resource isn't defined
      for i in range (len(fault_types)) :
         test= fault_types[i]

         if "ResourceExhaustion" in test and "FileManager" not in targets and "NetworkMonitor" not in targets :
           print ("Must include a target when specifying resources. For reference, these are your available targets")
           print (self.resource_limits)
           print ("Skipping: ")
           print(test)
           try :
              fault_types.remove(test) #prevent the crash if target is not set correctly
           except ValueError as e:
               print (f"Error is present with the test, continuing anyway")

      test_cases = list(itertools.product(fault_types, targets)) #list(set(list(itertools.product(fault_types, targets))))
      #Adding error handling to ensure that no key is null
      test_cases = [(ft,t) for (ft,t) in test_cases if ft and t]
      test_results = []
      for i, (fault_type, target) in enumerate(test_cases):
          print(f"\n--- Test Case {i + 1}/{len(test_cases)}: Fault={fault_type}, Target={target} ---")

          # Reset simulation state before each test case
          self.recovery_performed = 0
          self.faults_injected = 0
          self.intermittent_failure_rate = 0.0
          self.keys = {
            'Hardware': 'hwkey',
            'Kernel': 'kkey',
            'Applications': 'appkey'
          }

          # Re-run passive scan and plan unlock order
          self.passive_scan(symbolic_system)
          unlock_plan = self.plan_unlock_order(symbolic_system)

          # Inject the fault
          limit = 100 #hard set so fault injection will have a target
          if (fault_type == "ResourceExhaustion"):
            if (target in self.resource_limits):
               fault_success = self.inject_fault(target, fault_type, limit)
            else: #skip if inject fails (target resource does not exist)
              continue
          else :
              fault_success = self.inject_fault(target, fault_type)

          if(fault_success) :
                # Execute Plan with Fault
              results = self.execute_plan(unlock_plan)

              # Generate Report
              report = self.generate_report()
              test_results.append(report)

      self.print_test_summary(test_results)

    def print_test_summary(self, test_results):
        print("\n--- Test Summary ---")
        total_test_cases = len(test_results)
        successful_test_cases = 0

        for i, report in enumerate(test_results):
            print(f"\n--- Test Case {i + 1}/{total_test_cases} ---")
            print(f"Faults Injected: {report['Faults Injected']}")
            print(f"Recovery Performed: {report['Recovery Performed']}")
            print(f"Success Rate: {report['Success Rate']}")

            if report['Success Rate'] == "100.00%":
                successful_test_cases += 1