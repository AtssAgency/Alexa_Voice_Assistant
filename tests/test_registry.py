"""Tests for service registry and discovery utilities"""

import pytest
from services.registry import (
    ServiceRegistry, ServiceType, ServiceInfo, ServiceClient, ServiceDiscovery
)


class TestServiceRegistry:
    """Test service registry functionality"""
    
    def test_service_registration(self):
        """Test registering services in the registry"""
        registry = ServiceRegistry()
        
        # Register a service
        service_info = registry.register_service("test_kwd", ServiceType.KWD)
        
        assert service_info.name == "test_kwd"
        assert service_info.service_type == ServiceType.KWD
        assert service_info.port == 5003  # Default KWD port
        assert service_info.host == "localhost"
        assert service_info.address == "localhost:5003"
    
    def test_service_registration_custom_port(self):
        """Test registering services with custom ports"""
        registry = ServiceRegistry()
        
        service_info = registry.register_service("test_service", ServiceType.LLM, port=8080)
        
        assert service_info.port == 8080
        assert service_info.address == "localhost:8080"
    
    def test_service_retrieval(self):
        """Test retrieving registered services"""
        registry = ServiceRegistry()
        
        # Register multiple services
        registry.register_service("kwd_service", ServiceType.KWD)
        registry.register_service("stt_service", ServiceType.STT)
        registry.register_service("llm_service", ServiceType.LLM)
        
        # Test get_service
        kwd_service = registry.get_service("kwd_service")
        assert kwd_service is not None
        assert kwd_service.service_type == ServiceType.KWD
        
        # Test get_services_by_type
        kwd_services = registry.get_services_by_type(ServiceType.KWD)
        assert len(kwd_services) == 1
        assert kwd_services[0].name == "kwd_service"
        
        # Test list_services
        all_services = registry.list_services()
        assert len(all_services) == 3
    
    def test_service_unregistration(self):
        """Test unregistering services"""
        registry = ServiceRegistry()
        
        registry.register_service("test_service", ServiceType.TTS)
        assert registry.get_service("test_service") is not None
        
        # Unregister the service
        result = registry.unregister_service("test_service")
        assert result is True
        assert registry.get_service("test_service") is None
        
        # Try to unregister non-existent service
        result = registry.unregister_service("non_existent")
        assert result is False
    
    def test_default_ports(self):
        """Test that default ports are correctly assigned"""
        registry = ServiceRegistry()
        
        expected_ports = {
            ServiceType.LOGGER: 5001,
            ServiceType.LOADER: 5002,
            ServiceType.KWD: 5003,
            ServiceType.STT: 5004,
            ServiceType.LLM: 5005,
            ServiceType.TTS: 5006,
        }
        
        for service_type, expected_port in expected_ports.items():
            service_info = registry.register_service(
                f"test_{service_type.value}", service_type
            )
            assert service_info.port == expected_port


class TestServiceDiscovery:
    """Test service discovery functionality"""
    
    def test_startup_order(self):
        """Test that startup order is correct"""
        registry = ServiceRegistry()
        discovery = ServiceDiscovery(registry)
        
        startup_order = discovery.get_startup_order()
        expected_order = [
            ServiceType.LOGGER,
            ServiceType.TTS,
            ServiceType.LLM,
            ServiceType.STT,
            ServiceType.KWD,
            ServiceType.LOADER,
        ]
        
        assert startup_order == expected_order
    
    def test_shutdown_order(self):
        """Test that shutdown order is reverse of startup"""
        registry = ServiceRegistry()
        discovery = ServiceDiscovery(registry)
        
        startup_order = discovery.get_startup_order()
        shutdown_order = discovery.get_shutdown_order()
        
        assert shutdown_order == list(reversed(startup_order))
    
    def test_dependency_validation(self):
        """Test service dependency validation"""
        registry = ServiceRegistry()
        discovery = ServiceDiscovery(registry)
        
        # Empty registry should fail validation
        is_valid, errors = discovery.validate_service_dependencies()
        assert not is_valid
        assert len(errors) > 0
        
        # Register all required services
        for service_type in discovery.get_startup_order():
            registry.register_service(f"test_{service_type.value}", service_type)
        
        # Now validation should pass
        is_valid, errors = discovery.validate_service_dependencies()
        assert is_valid
        assert len(errors) == 0
    
    def test_services_in_startup_order(self):
        """Test getting services in startup order"""
        registry = ServiceRegistry()
        discovery = ServiceDiscovery(registry)
        
        # Register services in random order
        registry.register_service("kwd_service", ServiceType.KWD)
        registry.register_service("logger_service", ServiceType.LOGGER)
        registry.register_service("tts_service", ServiceType.TTS)
        
        # Get services in startup order
        ordered_services = discovery.get_services_in_startup_order()
        
        # Should be ordered: LOGGER, TTS, KWD (only registered ones)
        assert len(ordered_services) == 3
        assert ordered_services[0].service_type == ServiceType.LOGGER
        assert ordered_services[1].service_type == ServiceType.TTS
        assert ordered_services[2].service_type == ServiceType.KWD


class TestServiceClient:
    """Test service client functionality"""
    
    def test_client_creation(self):
        """Test creating a service client"""
        registry = ServiceRegistry()
        client = ServiceClient(registry)
        
        assert client.registry is registry
        assert len(client._channels) == 0
    
    def test_channel_management(self):
        """Test gRPC channel management"""
        registry = ServiceRegistry()
        client = ServiceClient(registry)
        
        # Register a service
        registry.register_service("test_service", ServiceType.KWD)
        
        # Get channel (this will create it)
        channel = client.get_channel("test_service")
        assert channel is not None
        assert len(client._channels) == 1
        
        # Get same channel again (should reuse)
        channel2 = client.get_channel("test_service")
        assert channel is channel2
        assert len(client._channels) == 1
        
        # Close channel
        client.close_channel("test_service")
        assert len(client._channels) == 0
    
    def test_nonexistent_service_channel(self):
        """Test getting channel for non-existent service"""
        registry = ServiceRegistry()
        client = ServiceClient(registry)
        
        channel = client.get_channel("non_existent")
        assert channel is None


class TestServiceInfo:
    """Test ServiceInfo dataclass"""
    
    def test_service_info_creation(self):
        """Test creating ServiceInfo objects"""
        info = ServiceInfo(
            name="test_service",
            service_type=ServiceType.LLM,
            host="localhost",
            port=5005
        )
        
        assert info.name == "test_service"
        assert info.service_type == ServiceType.LLM
        assert info.host == "localhost"
        assert info.port == 5005
        assert info.address == "localhost:5005"
        assert info.healthy is False
        assert info.last_health_check is None